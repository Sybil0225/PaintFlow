import re
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import cv2
import torchvision.transforms as transforms


def load_split(split_file):
    """Read a CoPaint split file (one id per line, e.g. '0304' or '00012.jpg'); returns a set of ids without extension."""
    if not split_file:
        return None
    with open(split_file) as f:
        return {os.path.splitext(l.strip())[0] for l in f if l.strip()}


def _vid(x):
    """'0304', '0304.mp4' or '/any/path/0304.mp4' -> '0304'"""
    return os.path.splitext(os.path.basename(str(x)))[0]


class PaintVideo(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        sample_size: int = 256,
        sample_stride: int = 1,
        sample_n_frames: int = 16,
        is_image: bool = False,
        transform=None,
        enhance_early_stages=True,
        early_stage_frames=4,
        split_file=None,
        **kwargs

    ):
        self.csv_path = csv_path
        self.video_folder = video_folder
        self.sample_size = sample_size
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        self.transform = transform
        self.enhance_early_stages = enhance_early_stages
        self.early_stage_frames = early_stage_frames

        # 加载 CSV 文件，并强制将 videoid 列读取为字符串
        assert os.path.exists(csv_path), f"找不到 CSV 文件：{csv_path}"
        self.data = pd.read_csv(self.csv_path, dtype={'videoid': str})  # 使用 dtype 参数保留前导零
        self.data['videoid'] = self.data['videoid'].apply(_vid)
        split = load_split(split_file)
        if split is not None:
            self.data = self.data[self.data['videoid'].isin(split)].reset_index(drop=True)

        # 生成视频路径列，拼接为完整路径，例如 "0001" -> "/workspace/test/videos/0001.mp4"
        self.data['video_path'] = self.data['videoid'].apply(lambda x: os.path.join(self.video_folder, f"{x}.mp4"))

        # 检查视频路径是否存在
        for video_path in self.data['video_path']:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在：{video_path}")

        print(f"成功加载 {len(self.data)} 个视频样本")

    def __len__(self):
        return len(self.data)

    def _enhance_line_art_frame(self, pil_image):
        """
        对草稿和线稿阶段的帧进行增强，加强线条同时减少细节

        Args:
            pil_image: PIL Image对象

        Returns:
            Enhanced PIL Image
        """
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        # 转换为numpy数组
        img_np = np.array(pil_image)

        # 1. 先进行高斯模糊以减少细节噪声
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(img_cv, (3, 3), 1.0)  # 轻微模糊减少细节

        # 2. 转换为灰度图进行边缘处理
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 3. 使用自适应阈值提取主要线条结构
        # 增大block_size和C值以获得更粗的线条，减少细节
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            15, 3  # 增大参数以减少细节
        )

        # 4. 使用形态学操作进一步简化线条
        kernel = np.ones((3, 3), np.uint8)  # 增大kernel以简化细节
        # 先腐蚀再膨胀，去除细小噪点
        morph_img = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        # 再膨胀使线条更粗
        morph_img = cv2.dilate(morph_img, kernel, iterations=1)

        # 5. 边缘检测 - 使用更高的阈值减少细节边缘
        edges = cv2.Canny(gray, 80, 200, apertureSize=3)  # 提高阈值

        # 6. 线条细化和加强
        # 使用距离变换和骨架化来获得更清晰的主线条
        dist_transform = cv2.distanceTransform(morph_img, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 7. 创建最终的线条图
        # 结合形态学处理和边缘信息
        combined_lines = cv2.bitwise_or(morph_img, sure_fg)

        # 8. 转换回RGB并增强对比度
        if len(combined_lines.shape) == 2:
            combined_lines_rgb = cv2.cvtColor(combined_lines, cv2.COLOR_GRAY2RGB)
        else:
            combined_lines_rgb = cv2.cvtColor(combined_lines, cv2.COLOR_BGR2RGB)

        enhanced_pil = Image.fromarray(combined_lines_rgb)

        # 9. 最终的对比度和锐度增强
        # 增强对比度使线条更突出
        contrast_enhancer = ImageEnhance.Contrast(enhanced_pil)
        enhanced_pil = contrast_enhancer.enhance(1.5)

        # 轻微锐化主要结构
        final_enhanced = enhanced_pil.filter(ImageFilter.UnsharpMask(
            radius=2, percent=120, threshold=5  # 调整参数专注于主要结构
        ))

        return final_enhanced

    def _extract_frames_from_video(self, video_path, sampled_indices=None):
        """
        从视频中提取帧，并对前几帧进行线条增强
        sampled_indices=None -> sample_n_frames uniformly spaced frames over the whole video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if sampled_indices is None:
            sampled_indices = np.linspace(0, total_frames - 1, self.sample_n_frames, dtype=int).tolist()
        sorted_indices = sorted(sampled_indices)

        for frame_idx in sorted_indices:
            if frame_idx >= total_frames:
                print(f"Warning: Frame index {frame_idx} exceeds video length {total_frames} for {video_path}")
                frame_idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                black_frame = np.zeros((self.sample_size, self.sample_size, 3), dtype=np.uint8)
                frames.append(Image.fromarray(black_frame))

        cap.release()

        # Reorder frames to match original sampled_indices order
        if sorted_indices != sampled_indices:
            index_mapping = {idx: i for i, idx in enumerate(sorted_indices)}
            reordered_frames = [frames[index_mapping[idx]] for idx in sampled_indices]
            frames = reordered_frames

        # 确保帧数正确
        if len(frames) != self.sample_n_frames:
            while len(frames) < self.sample_n_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (self.sample_size, self.sample_size)))
            frames = frames[:self.sample_n_frames]

        # 对前几帧进行线条增强
        if self.enhance_early_stages:
            enhanced_frames = []
            for i, frame in enumerate(frames):
                if i < self.early_stage_frames:
                    # 增强前几帧的线条
                    enhanced_frame = self._enhance_line_art_frame(frame)
                    enhanced_frames.append(enhanced_frame)

                    # 可选：保存增强前后对比图用于调试
                    if hasattr(self, '_debug_save') and self._debug_save and i < 2:
                        debug_dir = "/workspace/debug_enhanced_frames"
                        os.makedirs(debug_dir, exist_ok=True)
                        frame.save(f"{debug_dir}/original_frame_{i}.png")
                        enhanced_frame.save(f"{debug_dir}/enhanced_frame_{i}.png")
                else:
                    # 后续帧保持原样
                    enhanced_frames.append(frame)
            frames = enhanced_frames

        return frames

    def __getitem__(self, idx):
        """
        获取指定索引的视频及相应的文本提示
        """
        sample = self.data.iloc[idx]
        video_path = sample["video_path"]
        prompt_text = sample["prompt"]

        if self.is_image:  # single frmae
            frames = self._extract_frames_from_video(video_path, [0])  # 提取第一帧
            frame = frames[0]
            return {
                "pixel_values": self._process_frame(frame),
                "text": prompt_text,
            }

        # video mode
        frames = self._extract_frames_from_video(video_path)  # 不传入sampled_indices，让函数内部生成
        return {
            "pixel_values": torch.stack([self._process_frame(frame) for frame in frames]),
            "text": prompt_text,
        }

    def _load_frames(self, video_path):
        """
        从指定视频路径采样连续帧
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 确保帧数足够
        if total_frames < self.sample_n_frames:
            raise ValueError(f"视频 {video_path} 帧数不足 {self.sample_n_frames}，仅包含 {total_frames} 帧")

        # 随机选择起始帧
        start_frame = random.randint(0, max(0, total_frames - self.sample_n_frames * self.sample_stride))

        # 采样帧
        frames = []
        for i in range(start_frame, start_frame + self.sample_n_frames * self.sample_stride, self.sample_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success:
                break  # 视频结尾
            frames.append(frame[..., ::-1])  # BGR -> RGB

        cap.release()

        if len(frames) < self.sample_n_frames:
            raise ValueError(f"无法从视频 {video_path} 中提取到足够的帧")

        return frames

    def _load_random_frame(self, video_path):
        """
        从视频中随机加载一帧
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"无法从视频 {video_path} 加载帧")

        return frame[..., ::-1]  # BGR -> RGB

    def _process_frame(self, frame):
        """
        数据预处理操作，保持宽高比并padding到正方形
        """
        # 转换为PIL图像
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame

        # 计算padding以保持宽高比
        w, h = pil_image.size
        target_size = self.sample_size

        # 计算缩放比例
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        # 创建正方形画布并居中粘贴
        square_image = Image.new('RGB', (target_size, target_size), color=(0, 0, 0))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        square_image.paste(pil_image, (paste_x, paste_y))

        # 应用其他变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        return transform(square_image)


class PaintCtrlDataset(Dataset):
    def __init__(self, video_dir, annotations_path, sketch2video_path=None, condition_type="rgb", transform=None,
                 resolution=(512, 512), split_file=None, max_groups_per_video=2):
        """
        Args:
            video_dir (str): Path to the directory containing video files.
            annotations_path (str): Path to the CSV file containing annotations (videoid, prompt).
            sketch2video_path (str, optional): Path to JSONL file mapping sketches to videos.
            condition_type (str): "auto", "rgb", or "sketch".
            transform (callable, optional): Optional transform to be applied on a sample.
            resolution (tuple): Desired resolution of frames (height, width).
        """
        self.video_dir = video_dir
        self.annotations = pd.read_csv(annotations_path)
        self.condition_type = condition_type
        self.transform = transform
        self.resolution = resolution
        self.sketch2video_data = []
        self.max_groups_per_video = max_groups_per_video
        split = load_split(split_file)
        if sketch2video_path:
            with open(sketch2video_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    # CoPaint release stores video_path relative to video_dir ('0304.mp4');
                    # old server jsonl used absolute /workspace paths -> keep only the file name
                    entry["video_path"] = os.path.basename(entry.get("video_path") or entry["videoid"])
                    if split is None or _vid(entry["video_path"]) in split:
                        self.sketch2video_data.append(entry)
        self.video_sample_mapping = self._prepare_samples()

    def _prepare_samples(self):
        """Create a mapping between global index and (video index, group index) pairs."""
        mapping = []
        for video_idx, video_data in enumerate(self.sketch2video_data):
            num_groups = len(video_data["target_indices"]) // 16
            num_groups = max(1, min(num_groups, self.max_groups_per_video))
            for group_idx in range(num_groups):
                mapping.append((video_idx, group_idx))
        return mapping

    def __len__(self):
        return len(self.video_sample_mapping)

    def __getitem__(self, idx):
        video_idx, group_idx = self.video_sample_mapping[idx]
        video_data = self.sketch2video_data[video_idx]
        video_path = video_data["video_path"]

        # 计算需要的帧索引（控制帧 + 目标帧 16 个）
        start_idx = group_idx * 16
        end_idx = min(start_idx + 16, len(video_data["target_indices"]))
        target_indices = list(video_data["target_indices"][start_idx:end_idx])  # copy: padding below must not mutate the cache
        while len(target_indices) < 16:
            target_indices.append(target_indices[-1])

        if "rgb_index" in video_data:
            control_index = video_data["rgb_index"]
        elif "sketch_index" in video_data:
            control_index = video_data["sketch_index"]
        else:
            control_index = video_data["target_indices"][0]

        needed = sorted(set([control_index] + target_indices))

        # 一次性遍历视频，只提取 needed 帧
        frames_dict = self._load_frames_subset(os.path.join(self.video_dir, video_path), needed)

        # 组装控制帧
        control_frame = frames_dict[control_index]
        if self.condition_type != "rgb":
            control_frame = self._process_sketch_to_black_white(control_frame)
        if self.transform:
            control_frame = self.transform(control_frame)
        control_tensor = torch.from_numpy(control_frame).permute(2, 0, 1).float() / 255.0
        control_tensor = control_tensor * 2.0 - 1.0

        # 组装目标帧（按 target_indices 顺序）
        target_frames = []
        for fi in target_indices:
            fr = frames_dict[fi]
            if self.transform:
                fr = self.transform(fr)
            target_frames.append(torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0)
        target_frames_tensor = torch.stack(target_frames, dim=0)  # [F, C, H, W]

        return {"sketch": control_tensor, "video": target_frames_tensor}

    def _load_frames_subset(self, video_path, needed_indices):
        """一次读取视频，只保留 needed_indices 那些帧；返回 {idx: frame_rgb_resized}"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get total frames and build clamped index mapping
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"No frames could be loaded from the video: {video_path}")

        needed_originals = list(needed_indices)
        # Clamp requested indices to valid range [0, total_frames-1]
        original_to_clamped = {orig: min(max(orig, 0), total_frames - 1) for orig in needed_originals}

        # Reverse map: clamped index -> list of original indices that require this frame
        clamped_to_originals = {}
        for orig, clamped in original_to_clamped.items():
            clamped_to_originals.setdefault(clamped, []).append(orig)

        needed_clamped = set(original_to_clamped.values())

        out = {}  # maps original index -> frame
        idx = 0
        target_h, target_w = self.resolution

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in needed_clamped:
                frame_resized = self._resize_with_aspect_ratio(frame, (target_h, target_w))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Fill all original indices that map to this clamped idx
                for orig in clamped_to_originals.get(idx, []):
                    out[orig] = frame_rgb
                # Early exit if we've satisfied all requested originals
                if len(out) == len(needed_originals):
                    break
            idx += 1

        cap.release()

        # If some originals are still missing (e.g., we never hit the last frame due to an early read failure),
        # try to fetch the last valid frame explicitly and fill those.
        if len(out) != len(needed_originals):
            missing_originals = [orig for orig in needed_originals if orig not in out]
            if missing_originals:
                # Read the last frame once if any missing map to it
                need_last = any(original_to_clamped[orig] == (total_frames - 1) for orig in missing_originals)
                last_frame_rgb = None
                if need_last:
                    cap2 = cv2.VideoCapture(video_path)
                    if cap2.isOpened():
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                        ret, frame = cap2.read()
                        if ret:
                            frame_resized = self._resize_with_aspect_ratio(frame, (target_h, target_w))
                            last_frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        cap2.release()
                # Fill what we can (currently only last frame fallback is needed for clamped requests)
                for orig in missing_originals:
                    clamped = original_to_clamped[orig]
                    if clamped == total_frames - 1 and last_frame_rgb is not None:
                        out[orig] = last_frame_rgb

        # As a final guard, if still missing, do not crash—use a black frame to keep training running.
        if len(out) != len(needed_originals):
            remaining = [orig for orig in needed_originals if orig not in out]
            black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for orig in remaining:
                out[orig] = black

        return out

    def _load_video_frame(self, video_path, frame_index):
        full_path = os.path.join(self.video_dir, video_path)
        video_frames = self._load_video_frames(full_path)
        return video_frames[frame_index]

    def _load_video_frames(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-allocate list for better performance
        frames = [None] * total_frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = self._resize_with_aspect_ratio(frame, self.resolution)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames[frame_count] = frame_rgb
            frame_count += 1

        cap.release()

        # Remove None entries and return actual frames
        frames = frames[:frame_count]

        if not frames:
            raise ValueError(f"No frames could be loaded from the video: {video_path}")
        return frames


    def _resize_with_aspect_ratio(self, frame, target_resolution):
        """
        Args:
            frame (numpy.ndarray): input frame [H, W, C]
            target_resolution (tuple): target resolution (height, width)

        """
        h, w = frame.shape[:2]
        target_h, target_w = target_resolution
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        frame_scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        final_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        final_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = frame_scaled

        return final_frame

    def _process_sketch_to_black_white(self, frame):
        """
        Convert a sketch frame to black-background with white lines.

        Args:
            frame (numpy.ndarray): Input sketch frame (assumed to be white background, black lines).

        Returns:
            numpy.ndarray: Sketch with black background and white lines.
        """
        # Convert to grayscale if it is in RGB format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Invert colors to make black background and white lines
        mean_value = np.mean(gray)
        if mean_value > 127:
            black_background_sketch = cv2.bitwise_not(gray)
        else:
            black_background_sketch = gray
        # Resize the sketch to the desired resolution
        black_background_sketch = cv2.resize(black_background_sketch, self.resolution)
        _, black_background_sketch = cv2.threshold(black_background_sketch, 50, 255, cv2.THRESH_BINARY)

        # Convert to 3-channel RGB format for compatibility
        black_background_sketch_rgb = np.stack([black_background_sketch] * 3, axis=-1)

        return black_background_sketch_rgb


# Function to visualize a few sketches
def save_processed_sketches(dataset, save_dir, indices):
    """
    Save processed sketches from the dataset into a specified folder.

    Args:
        dataset (PaintCtrlDataset): The dataset object.
        save_dir (str): Directory to save the processed sketches.
        indices (list): List of indices of sketches to save.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        # Fetch processed sketch from the dataset
        sample = dataset[idx]
        sketch_tensor = sample["sketch"]  # Shape: [C, H, W]

        # Convert PyTorch tensor to NumPy array for saving
        sketch_np = sketch_tensor.permute(1, 2, 0).numpy()  # [H, W, C]

        # Scale the sketch back to [0, 255] range for saving
        sketch_np = (sketch_np * 255).astype(np.uint8)

        # Save the sketch using OpenCV
        save_path = os.path.join(save_dir, f"sketch_{idx}.png")
        cv2.imwrite(save_path, sketch_np)

        print(f"Saved sketch {idx} to: {save_path}")



class ContinuousLineExtractor:
    """
    连续线条提取器 - 白底黑线，细线条，保持线条连续性
    """

    def __init__(self, line_thickness=1, noise_reduction=True, line_connection=True):
        """
        Args:
            line_thickness: 线条粗细
            noise_reduction: 是否减少噪声
            line_connection: 是否连接断开的线条
        """
        self.line_thickness = line_thickness
        self.noise_reduction = noise_reduction
        self.line_connection = line_connection

    def extract_continuous_lines(self, pil_image, method='connected_canny'):
        """
        提取连续线条，避免断续

        Args:
            pil_image: PIL Image对象
            method: 提取方法
                - 'connected_canny': 连接的Canny边缘 (推荐)
                - 'line_linking': 线条连接算法
                - 'gap_filling': 填充间隙
                - 'smooth_lines': 平滑线条
        """
        if method == 'connected_canny':
            return self._connected_canny_extraction(pil_image)
        elif method == 'line_linking':
            return self._line_linking_extraction(pil_image)
        elif method == 'gap_filling':
            return self._gap_filling_extraction(pil_image)
        elif method == 'smooth_lines':
            return self._smooth_lines_extraction(pil_image)
        else:
            return self._connected_canny_extraction(pil_image)

    def _connected_canny_extraction(self, pil_image):
        """
        连接的Canny边缘检测 - 主动连接断开的线条
        """
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. 轻微预处理，保持边缘信息
        if self.noise_reduction:
            # 使用保边降噪
            denoised = cv2.bilateralFilter(gray, 3, 50, 50)
        else:
            denoised = gray

        # 2. 双阈值Canny检测
        # 较低的低阈值帮助连接弱边缘
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(denoised, low_threshold, high_threshold, apertureSize=3, L2gradient=True)

        # 3. 连接断开的线条
        if self.line_connection:
            edges = self._connect_broken_lines(edges)

        # 4. 最后清理 - 只去除明显的噪声点，不影响线条连续性
        if self.noise_reduction:
            edges = self._gentle_noise_removal(edges)

        return self._create_white_background_image(edges)

    def _line_linking_extraction(self, pil_image):
        """
        线条连接算法 - 使用HoughLines检测并连接线段
        """
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. 边缘检测
        edges = cv2.Canny(gray, 60, 180)

        # 2. 检测线段
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=10, maxLineGap=15)  # 增大maxLineGap连接更多线段

        # 3. 在新图像上绘制连接的线条
        line_image = np.zeros_like(edges)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, self.line_thickness)

        # 4. 补充原始边缘信息
        combined = cv2.bitwise_or(line_image, edges)

        # 5. 进一步连接
        if self.line_connection:
            combined = self._connect_nearby_endpoints(combined)

        return self._create_white_background_image(combined)

    def _gap_filling_extraction(self, pil_image):
        """
        间隙填充提取 - 专门填充线条间的小间隙
        """
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. 基础边缘检测
        edges = cv2.Canny(gray, 60, 160)

        # 2. 形态学操作填充小间隙
        # 使用线性核连接水平和垂直方向的线条

        # 水平连接
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)

        # 垂直连接
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)

        # 斜向连接
        diagonal_kernel1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        diagonal_kernel2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        diagonal_connected1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, diagonal_kernel1)
        diagonal_connected2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, diagonal_kernel2)

        # 3. 组合所有方向的连接
        connected = cv2.bitwise_or(horizontal_connected, vertical_connected)
        connected = cv2.bitwise_or(connected, diagonal_connected1)
        connected = cv2.bitwise_or(connected, diagonal_connected2)

        # 4. 自定义细化保持线条粗细
        if self.line_thickness == 1:
            connected = self._custom_thinning(connected)

        return self._create_white_background_image(connected)

    def _smooth_lines_extraction(self, pil_image):
        """
        平滑线条提取 - 创建更平滑的连续线条
        """
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. 轻微模糊创建更连续的边缘
        slightly_blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)

        # 2. 梯度幅度计算
        grad_x = cv2.Sobel(slightly_blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(slightly_blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 3. 自适应阈值创建连续边缘
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, smooth_edges = cv2.threshold(magnitude_norm, 30, 255, cv2.THRESH_BINARY)

        # 4. 温和的连接操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smooth_connected = cv2.morphologyEx(smooth_edges, cv2.MORPH_CLOSE, kernel)

        # 5. 自定义细化到单像素
        if self.line_thickness == 1:
            smooth_connected = self._custom_thinning(smooth_connected)

        return self._create_white_background_image(smooth_connected)

    def _connect_broken_lines(self, binary_image, max_gap=5):
        """
        连接断开的线条
        """
        # 使用不同方向的核连接断开的线条
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap, 1)),  # 水平
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max_gap)),  # 垂直
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),  # 小范围连接
        ]

        connected = binary_image.copy()
        for kernel in kernels:
            connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel)

        return connected

    def _connect_nearby_endpoints(self, binary_image, max_distance=8):
        """
        连接附近的端点
        """
        # 找到线条的端点
        endpoints = self._find_endpoints(binary_image)

        result = binary_image.copy()

        # 连接距离较近的端点
        for i, (x1, y1) in enumerate(endpoints):
            for j, (x2, y2) in enumerate(endpoints[i + 1:], i + 1):
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance <= max_distance:
                    cv2.line(result, (x1, y1), (x2, y2), 255, 1)

        return result

    def _find_endpoints(self, binary_image):
        """
        找到线条的端点
        """
        # 使用3x3窗口检测端点（只有一个或两个邻居的点）
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0

        # 计算每个点的邻居数量
        neighbors = cv2.filter2D(binary_image // 255, -1, kernel)

        # 端点：有1个邻居的点
        endpoints_mask = (binary_image > 0) & (neighbors == 1)

        # 获取端点坐标
        y_coords, x_coords = np.where(endpoints_mask)
        endpoints = list(zip(x_coords, y_coords))

        return endpoints

    def _gentle_noise_removal(self, binary_image, min_area=8):
        """
        温和的噪声去除 - 不会断开线条
        """
        # 只去除非常小的连通区域
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_image = binary_image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                # 检查是否是孤立的噪声点（没有连接到其他线条）
                mask = np.zeros_like(binary_image)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                # 膨胀一点看是否连接到其他线条
                dilated_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
                overlap = cv2.bitwise_and(binary_image, dilated_mask)

                # 如果膨胀后没有明显重叠，认为是噪声
                if cv2.countNonZero(overlap) <= cv2.countNonZero(mask) + 5:
                    cv2.drawContours(clean_image, [contour], -1, 0, -1)

        return clean_image

    def _custom_thinning(self, binary_image):
        """
        自定义细化算法 - 替代cv2.ximgproc.thinning
        """
        # 使用简单的骨架化算法
        return self._skeletonize(binary_image)

    def _skeletonize(self, binary_img):
        """
        骨架化算法 - 得到单像素宽度的线条
        """
        skeleton = np.zeros(binary_img.shape, np.uint8)
        eroded = binary_img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(eroded, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            eroded = cv2.erode(eroded, kernel)

            if cv2.countNonZero(eroded) == 0:
                break

        return skeleton

    def _create_white_background_image(self, line_mask):
        """
        创建白底黑线图像
        """
        # 创建白色背景
        white_background = np.full((*line_mask.shape, 3), 255, dtype=np.uint8)

        # 设置线条为黑色
        line_positions = line_mask > 0
        white_background[line_positions] = [0, 0, 0]

        return Image.fromarray(white_background)


class PaintDataset(Dataset):
    def __init__(
            self,
            video_json_path,
            video_folder,
            csv_path,
            sample_size=256,
            sample_n_frames=16,
            transform=None,
            enhance_early_stages=True,
            early_stage_frames=4,
            line_extraction_method='connected_canny',  # 连续线条提取
            line_connection=True,  # 启用线条连接
            noise_reduction=True,
            split_file=None,
            **kwargs
    ):
        """
        PaintDataset - 连续线条版本，解决断续问题

        Args:
            line_extraction_method: 线条提取方法
                - 'connected_canny': 连接的Canny边缘 (推荐)
                - 'line_linking': HoughLines线条连接
                - 'gap_filling': 间隙填充算法
                - 'smooth_lines': 平滑线条提取
            line_connection: 是否启用线条连接
        """
        self.video_folder = video_folder
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.transform = transform

        # 连续线条设置
        self.enhance_early_stages = enhance_early_stages
        self.early_stage_frames = early_stage_frames
        self.line_extraction_method = line_extraction_method
        self.line_connection = line_connection
        self.noise_reduction = noise_reduction

        # 初始化连续线条提取器
        self.line_extractor = ContinuousLineExtractor(
            line_thickness=1,
            noise_reduction=noise_reduction,
            line_connection=line_connection
        )

        # Load data
        self.video_data = []
        split = load_split(split_file)
        with open(video_json_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                # accept both {"videoid": "0304.mp4"} (release) and {"video_path": "/abs/0304.mp4"} (old)
                data['videoid'] = _vid(data.get('videoid') or data['video_path']) + '.mp4'
                if split is None or _vid(data['videoid']) in split:
                    self.video_data.append(data)
        random.shuffle(self.video_data)

        self.annotations = pd.read_csv(csv_path, dtype={'videoid': str})

        # Create prompt mapping
        self.prompt_mapping = {}
        jsonl_videoids = set(item['videoid'] for item in self.video_data)

        matched_count = 0
        for _, row in self.annotations.iterrows():
            csv_videoid = str(row.get('videoid', ''))
            prompt = str(row.get('prompt', ''))

            if csv_videoid in jsonl_videoids:
                self.prompt_mapping[csv_videoid] = prompt
                matched_count += 1
            else:
                if f"{csv_videoid}.mp4" in jsonl_videoids:
                    self.prompt_mapping[f"{csv_videoid}.mp4"] = prompt
                    matched_count += 1
                elif csv_videoid.endswith('.mp4'):
                    base_name = csv_videoid[:-4]
                    if base_name in jsonl_videoids:
                        self.prompt_mapping[base_name] = prompt
                        matched_count += 1

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((sample_size, sample_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        print(f"=== PaintDataset 连续线条模式 (无ximgproc依赖) ===")
        print(f"视频样本数量: {len(self.video_data)}")
        print(f"匹配的Prompt数量: {matched_count}")
        print(f"连续线条设置:")
        print(f"  ✓ enhance_early_stages: {self.enhance_early_stages}")
        print(f"  ✓ early_stage_frames: {self.early_stage_frames}")
        print(f"  ✓ line_extraction_method: {self.line_extraction_method}")
        print(f"  ✓ line_connection: {self.line_connection} (解决断续问题)")
        print(f"  ✓ 效果: 白底黑线，连续细线条，减少断续")
        print(f"====================================================")

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_info = self.video_data[idx]
        videoid = video_info['videoid']
        sampled_indices = video_info['sampled_indices']

        video_path = os.path.join(self.video_folder, videoid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 提取连续线条
        frames = self._extract_frames_with_continuous_lines(video_path, sampled_indices)

        # Apply transforms
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)

        video_tensor = torch.stack(transformed_frames)

        # Get prompt
        prompt = self.prompt_mapping.get(videoid, "")
        if not prompt:
            base_videoid = videoid.replace('.mp4', '') if videoid.endswith('.mp4') else f"{videoid}.mp4"
            prompt = self.prompt_mapping.get(base_videoid, "")
            if not prompt:
                prompt = "Painting process: sketch → line art → base colors → shading."

        sample = {
            'pixel_values': video_tensor,
            'text': prompt,
            'videoid': videoid,
            'sampled_indices': sampled_indices
        }

        return sample

    def _extract_frames_with_continuous_lines(self, video_path, sampled_indices):
        """
        提取连续线条帧
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sorted_indices = sorted(sampled_indices)

        # 提取所有帧
        for frame_idx in sorted_indices:
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                white_frame = np.full((self.sample_size, self.sample_size, 3), 255, dtype=np.uint8)
                frames.append(Image.fromarray(white_frame))

        cap.release()

        # 重新排序
        if sorted_indices != sampled_indices:
            index_mapping = {idx: i for i, idx in enumerate(sorted_indices)}
            reordered_frames = [frames[index_mapping[idx]] for idx in sampled_indices]
            frames = reordered_frames

        # 确保帧数正确
        if len(frames) != self.sample_n_frames:
            while len(frames) < self.sample_n_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (self.sample_size, self.sample_size), 'white'))
            frames = frames[:self.sample_n_frames]

        # 🎯 前四帧转换为连续线条
        if self.enhance_early_stages:
            processed_frames = []
            for i, frame in enumerate(frames):
                if i < self.early_stage_frames:
                    # 提取连续线条
                    continuous_line_frame = self.line_extractor.extract_continuous_lines(
                        frame, self.line_extraction_method
                    )
                    processed_frames.append(continuous_line_frame)

                    # 调试保存
                    if hasattr(self, '_debug_save') and self._debug_save and i < 2:
                        debug_dir = "/workspace/debug_continuous_lines"
                        os.makedirs(debug_dir, exist_ok=True)
                        frame.save(f"{debug_dir}/original_frame_{i}.png")
                        continuous_line_frame.save(
                            f"{debug_dir}/continuous_line_frame_{i}_{self.line_extraction_method}.png")
                else:
                    # 后续帧保持原样
                    processed_frames.append(frame)

            frames = processed_frames

        return frames

    def enable_debug_save(self):
        """启用调试保存"""
        self._debug_save = True
        print("Debug save enabled. Continuous line frames will be saved to /workspace/debug_continuous_lines/")

    def test_continuous_lines(self, num_samples=3, output_dir="/workspace/continuous_line_test"):
        """
        测试连续线条效果
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"测试连续线条效果，解决断续问题，保存到: {output_dir}")

        for i in range(min(num_samples, len(self))):
            try:
                sample = self[i]
                video_tensor = sample['pixel_values']
                videoid = sample['videoid']

                test_dir = os.path.join(output_dir, f"test_{i:02d}_{videoid.split('.')[0]}")
                os.makedirs(test_dir, exist_ok=True)

                for j in range(video_tensor.shape[0]):
                    frame_tensor = video_tensor[j]
                    frame_np = ((frame_tensor + 1.0) / 2.0).clamp(0, 1)
                    frame_np = (frame_np * 255).byte().permute(1, 2, 0).numpy()
                    frame_pil = Image.fromarray(frame_np)

                    if j < self.early_stage_frames and self.enhance_early_stages:
                        frame_name = f"frame_{j:02d}_CONTINUOUS_LINES_{self.line_extraction_method}.jpg"
                    else:
                        frame_name = f"frame_{j:02d}_ORIGINAL.jpg"

                    frame_pil.save(os.path.join(test_dir, frame_name))

                # 保存信息
                with open(os.path.join(test_dir, "continuous_line_info.txt"), "w") as f:
                    f.write(f"Video ID: {videoid}\n")
                    f.write(f"Continuous line extraction: {self.enhance_early_stages}\n")
                    f.write(f"Line connection enabled: {self.line_connection}\n")
                    f.write(f"Method: {self.line_extraction_method}\n")
                    f.write(f"改进特点:\n")
                    f.write(f"  - 解决线条断续问题\n")
                    f.write(f"  - 主动连接断开的线条\n")
                    f.write(f"  - 填充小间隙\n")
                    f.write(f"  - 保持线条连续性\n")
                    f.write(f"  - 移除了cv2.ximgproc依赖\n")

                print(f"✓ Continuous line test {i} completed: {test_dir}")

            except Exception as e:
                print(f"✗ Error testing sample {i}: {e}")

        print(f"连续线条测试完成! 查看结果: {output_dir}")


class PaintStaticDataset(Dataset):
    """PaintStatic stage-labelled image-text pairs.
    captions.json: [{"image": "00012.jpg", "text": "[s0:Sketch] <caption>"}]
    Returns pixel_values in [-1, 1], the caption text and stage_idx (0 Sketch, 1 Line, 2 BaseColor, 3 LightShadow)."""
    STAGE_PATTERN = re.compile(r"\[s(\d+):[^\]]+\]")

    def __init__(self, captions_file, images_dir, sample_size=512, split_file=None, transform=None):
        with open(captions_file, encoding="utf-8") as f:
            items = json.load(f)
        split = load_split(split_file)
        self.items = [x for x in items if split is None or _vid(x["image"]) in split]
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(sample_size), transforms.CenterCrop(sample_size),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(os.path.join(self.images_dir, item["image"])).convert("RGB")
        m = self.STAGE_PATTERN.search(item["text"])
        return {"pixel_values": self.transform(image), "text": item["text"],
                "stage_idx": int(m.group(1)) if m else -1, "image": item["image"]}


if __name__ == "__main__":
    print("🎨 测试PaintDataset连续线条功能 (无ximgproc依赖)...")

    dataset = PaintDataset(
        video_json_path="/workspace/video_ctrl/video_16_enhance.jsonl",
        video_folder="/workspace/video_ctrl/videos",
        csv_path="/workspace/final_frame/annotation_ctrl.csv",
        sample_size=256,
        sample_n_frames=16,
        enhance_early_stages=True,
        early_stage_frames=4,
        line_extraction_method='connected_canny',  # 推荐
        line_connection=True,  # 启用线条连接
        noise_reduction=True
    )

    # 测试连续线条效果
    dataset.test_continuous_lines(num_samples=5)

    print("✅ 连续线条测试完成！解决了断续问题，无需ximgproc。")