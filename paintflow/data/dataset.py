import os, io, csv, math, random
from decord import VideoReader
from pathlib import Path
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            video_json_path=None,  # Add parameter for video_16.json
        ):
        self.video_json_path = video_json_path  # Store the video_16.json path
        self.video_json_data = None  # Initialize as None to maintain compatibility

        if video_json_path:  # If a JSON/JSONL path is provided, parse the file
            self.video_json_data = self._load_json_file(video_json_path)  # Parse JSON/JSONL file
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        # zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size

        # Pixel transforms (for image-based data)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def _load_json_file(self, file_path):
        """
        Helper function to load JSON/JSONL data.

        Args:
            file_path (str): Path to the file.

        Returns:
            dict: Parsed JSON data.
        """
        file_path = Path(file_path)
        if file_path.suffix == '.json':  # If the file is a regular JSON file
            with open(file_path, 'r') as jsonfile:
                return json.load(jsonfile)
        elif file_path.suffix == '.jsonl':  # If the file is a JSON-lines file
            json_data = {}
            with open(file_path, 'r') as jsonfile:
                for line in jsonfile:
                    obj = json.loads(line)
                    video_id = obj.get("videoid")  # Assuming "videoid" is the key linking to dataset
                    if video_id:
                        json_data[video_id] = obj
            return json_data
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, prompt = video_dict['videoid'], video_dict['prompt']

        # Check if sampled indices are available in video_16.json
        json_sampled_indices = None
        if self.video_json_data and videoid in self.video_json_data:
            json_sampled_indices = self.video_json_data[videoid]
        videoid, prompt = video_dict['videoid'], video_dict['prompt']
        # Construct the path to the video
        video_path    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        if not self.is_image:
            if json_sampled_indices:  # Use indices from video_16.json if available
                batch_index = np.array(json_sampled_indices[:self.sample_n_frames], dtype=int)
            else:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx   = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            if json_sampled_indices:  # Use the first index for images
                batch_index = [json_sampled_indices[0]]
            else:
                batch_index = [random.randint(0, video_length - 1)]

        # Read video frames and convert them to tensor
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, prompt

    def __len__(self):
        """
                Updated to include each 16-frame group in JSONL for each video.
                """
        # Videos * sets of frame samples (2 in this case)
        return self.length * 2 if self.video_json_data else self.length

    def __getitem__(self, idx):
        video_idx = idx // 2  # Determine the video index
        group_idx = idx % 2  # Determine which group of frames to sample

        # Retrieve the specific video
        video_dict = self.dataset[video_idx]
        videoid, prompt = video_dict['videoid'], video_dict['prompt']

        # Get the corresponding sampled indices for the group
        json_sampled_indices = None
        if self.video_json_data and videoid in self.video_json_data:
            json_sampled_indices = self.video_json_data[videoid][group_idx]

        # Construct the video path and load frames
        video_path = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        if not self.is_image:
            if json_sampled_indices:  # Use predefined indices
                batch_index = np.array(json_sampled_indices[:self.sample_n_frames], dtype=int)
            else:
                clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            # Fallback for images
            if json_sampled_indices:
                batch_index = [json_sampled_indices[0]]
            else:
                batch_index = [random.randint(0, video_length - 1)]

        # Read and process frames
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        pixel_values = self.pixel_transforms(pixel_values)

        # Return the sample
        sample = dict(pixel_values=pixel_values, text=prompt)
        return sample


    def save_first_10_samples(self):
        """保存前10条数据的16帧视频"""
        import os
        from PIL import Image

        save_dir = "/workspace/dataset_samples"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(min(10, len(self))):
            print(f"保存第 {i} 个样本...")

            # 获取数据
            sample = self[i]
            pixel_values = sample['pixel_values']  # [F, C, H, W]
            text = sample['text']

            # 创建样本文件夹
            sample_dir = os.path.join(save_dir, f"sample_{i:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            # 保存每一帧
            for frame_idx in range(pixel_values.shape[0]):
                frame = pixel_values[frame_idx]  # [C, H, W]

                # 转换为PIL图像 (假设数据是[-1,1]范围)
                frame = (frame + 1.0) / 2.0  # 转到[0,1]
                frame = (frame * 255).clamp(0, 255).byte()
                frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

                # 保存
                img = Image.fromarray(frame)
                img.save(os.path.join(sample_dir, f"frame_{frame_idx:02d}.png"))

            # 保存文本
            with open(os.path.join(sample_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"已保存到: {sample_dir}")

        print(f"所有样本保存完成，路径: {save_dir}")
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import cv2
import torchvision.transforms as transforms


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
    ):
        self.csv_path = csv_path
        self.video_folder = video_folder
        self.sample_size = sample_size
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        self.transform = transform

        # 加载 CSV 文件，并强制将 videoid 列读取为字符串
        assert os.path.exists(csv_path), f"找不到 CSV 文件：{csv_path}"
        self.data = pd.read_csv(self.csv_path, dtype={'videoid': str})  # 使用 dtype 参数保留前导零

        # 生成视频路径列，拼接为完整路径，例如 "0001" -> "/workspace/test/videos/0001.mp4"
        self.data['video_path'] = self.data['videoid'].apply(lambda x: os.path.join(self.video_folder, f"{x}.mp4"))

        # 检查视频路径是否存在
        for video_path in self.data['video_path']:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在：{video_path}")

        print(f"成功加载 {len(self.data)} 个视频样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的视频及相应的文本提示
        """
        sample = self.data.iloc[idx]
        video_path = sample["video_path"]
        prompt_text = sample["prompt"]

        if self.is_image:  # 单帧模式（如果被指定）
            frame = self._load_random_frame(video_path)
            return {
                "pixel_values": self._process_frame(frame),
                "text": prompt_text,
            }

        # 视频模式（加载多帧）
        frames = self._load_frames(video_path)
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
                 resolution=(512, 512)):
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
        if sketch2video_path:
            with open(sketch2video_path, 'r') as f:
                for line in f:
                    self.sketch2video_data.append(json.loads(line))
        self.video_sample_mapping = self._prepare_samples()

    def _prepare_samples(self):
        """Create a mapping between global index and (video index, group index) pairs."""
        mapping = []
        for video_idx, video_data in enumerate(self.sketch2video_data):
            num_groups = len(video_data["target_indices"]) // 16
            num_groups = min(num_groups, 2)
            for group_idx in range(num_groups):
                mapping.append((video_idx, group_idx))
        return mapping

    def __len__(self):
        return len(self.video_sample_mapping)

    def __getitem__(self, idx):
        video_idx, group_idx = self.video_sample_mapping[idx]
        video_data = self.sketch2video_data[video_idx]

        video_path = video_data["video_path"]
        sketch_index = video_data["sketch_index"]
        start_idx = group_idx * 16
        end_idx = min(start_idx + 16, len(video_data["target_indices"]))
        target_indices = video_data["target_indices"][start_idx:end_idx]
        while len(target_indices) < 16:
            target_indices.append(target_indices[-1])


        # Load and process control (sketch/rgb) frame
        sketch_frame = self._load_video_frame(video_path, sketch_index)
        if self.condition_type == "rgb":
            # rgb ctrl: keep original color info
            sketch_frame = sketch_frame
        else:
            # sketch ctrl: convert to black and white
            sketch_frame = self._process_sketch_to_black_white(sketch_frame)

        # Apply additional transformations if defined
        if self.transform:
            sketch_frame = self.transform(sketch_frame)

        # Convert sketch frame to PyTorch tensor
        sketch_tensor = torch.from_numpy(sketch_frame).permute(2, 0, 1).float() / 255.0
        sketch_tensor = sketch_tensor * 2.0 - 1.0

        # Load and preprocess target frames
        target_frames = [self._load_video_frame(video_path, i) for i in target_indices]
        if self.transform:
            target_frames = [self.transform(frame) for frame in target_frames]
        target_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in target_frames]
        target_frames_tensor = torch.stack(target_frames, dim=0)

        return {
            "sketch": sketch_tensor,
            "video": target_frames_tensor,
        }

    def _load_video_frame(self, video_path, frame_index):
        full_path = os.path.join(self.video_dir, video_path)
        video_frames = self._load_video_frames(full_path)
        return video_frames[frame_index]

    def _load_video_frames(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = self._resize_with_aspect_ratio(frame, self.resolution)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

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




class PaintDataset(Dataset):
    def __init__(
            self,
            video_json_path,
            video_folder,
            csv_path,
            sample_size=256,
            sample_n_frames=16,
            transform=None,
            **kwargs
    ):
        """
        PaintDataset for loading painting process videos with frame sampling

        Args:
            video_json_path: Path to the JSONL file containing video information
            video_folder: Path to the folder containing video files
            csv_path: Path to CSV file containing prompts for each video ID
            sample_size: Size to resize frames to
            sample_n_frames: Number of frames to extract (should be 16)
            transform: Optional transform to apply to frames
        """
        self.video_folder = video_folder
        self.sample_size = sample_size
        self.sample_n_frames = sample_n_frames
        self.transform = transform

        # Load video data from JSONL file
        self.video_data = []
        with open(video_json_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.video_data.append(data)
        random.shuffle(self.video_data)
        print(f"Shuffled {len(self.video_data)} video samples")

        # Load CSV annotations - assuming CSV has columns like 'videoid' and 'prompt'
        self.annotations = pd.read_csv(csv_path, dtype={'videoid': str}
)

        # Create a mapping from videoid to prompt for faster lookup
        self.prompt_mapping = {}
        jsonl_videoids = set(item['videoid'] for item in self.video_data)

        # 创建映射时进行灵活匹配
        matched_count = 0
        unmatched_csv_ids = []
        for _, row in self.annotations.iterrows():
            csv_videoid = str(row.get('videoid', ''))
            prompt = str(row.get('prompt', ''))

            # 首先尝试精确匹配
            if csv_videoid in jsonl_videoids:
                self.prompt_mapping[csv_videoid] = prompt
                matched_count += 1
            else:
                # 尝试添加.mp4扩展名
                if f"{csv_videoid}.mp4" in jsonl_videoids:
                    self.prompt_mapping[f"{csv_videoid}.mp4"] = prompt
                    matched_count += 1
                # 尝试移除.mp4扩展名
                elif csv_videoid.endswith('.mp4'):
                    base_name = csv_videoid[:-4]
                    if base_name in jsonl_videoids:
                        self.prompt_mapping[base_name] = prompt
                        matched_count += 1
                else:
                    unmatched_csv_ids.append(csv_videoid)

        print(f"Successfully matched {matched_count} videoids between CSV and JSONL")
        print(f"Total CSV entries: {len(self.annotations)}")
        print(f"Total JSONL entries: {len(self.video_data)}")

        if unmatched_csv_ids:
            print(f"\n未匹配的CSV videoid (前5个):")
            for i, vid in enumerate(unmatched_csv_ids[:5]):
                print(f"  未匹配: '{vid}'")

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((sample_size, sample_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        print(f"Loaded {len(self.video_data)} samples from {video_json_path}")
        print(f"Loaded {len(self.prompt_mapping)} prompts from {csv_path}")

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_info = self.video_data[idx]
        videoid = video_info['videoid']
        sampled_indices = video_info['sampled_indices']

        # Construct video file path
        video_path = os.path.join(self.video_folder, videoid)

        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract frames from video
        frames = self._extract_frames_from_video(video_path, sampled_indices)

        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)

        # Stack frames into a tensor [C, T, H, W] or [T, C, H, W] depending on your model
        video_tensor = torch.stack(transformed_frames)  # Shape: [T, C, H, W]

        # Get prompt for this video
        prompt = self.prompt_mapping.get(videoid, "")
        if not prompt:
            # 尝试备用匹配
            base_videoid = videoid.replace('.mp4', '') if videoid.endswith('.mp4') else f"{videoid}.mp4"
            prompt = self.prompt_mapping.get(base_videoid, "")

            if not prompt:
                # 只在第一次遇到时打印warning，避免重复输出
                if not hasattr(self, '_warned_ids'):
                    self._warned_ids = set()
                if videoid not in self._warned_ids:
                    print(f"Warning: No prompt found for videoid {videoid}")
                    self._warned_ids.add(videoid)
                # 使用默认prompt而不是空字符串
                prompt = "Painting process: sketch → line art → base colors → shading."

        # Prepare return dictionary
        sample = {
            'pixel_values': video_tensor,  # The video frames
            'text': prompt,  # The prompt text
            'videoid': videoid,  # Video ID for debugging
            'sampled_indices': sampled_indices  # Frame indices for debugging
        }

        return sample

    def _extract_frames_from_video(self, video_path, sampled_indices):
        """
        Extract frames from video at specified indices

        Args:
            video_path: Path to video file
            sampled_indices: List of frame indices to extract

        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sort indices to read frames in order (more efficient)
        sorted_indices = sorted(sampled_indices)

        for frame_idx in sorted_indices:
            # Check if frame index is valid
            if frame_idx >= total_frames:
                print(f"Warning: Frame index {frame_idx} exceeds video length {total_frames} for {video_path}")
                # Use last frame if index is out of bounds
                frame_idx = total_frames - 1

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                # Create a black frame as fallback
                black_frame = np.zeros((self.sample_size, self.sample_size, 3), dtype=np.uint8)
                frames.append(Image.fromarray(black_frame))

        cap.release()

        # Reorder frames to match original sampled_indices order
        if sorted_indices != sampled_indices:
            index_mapping = {idx: i for i, idx in enumerate(sorted_indices)}
            reordered_frames = [frames[index_mapping[idx]] for idx in sampled_indices]
            frames = reordered_frames

        # Ensure we have exactly sample_n_frames frames
        if len(frames) != self.sample_n_frames:
            print(f"Warning: Expected {self.sample_n_frames} frames, got {len(frames)} for {video_path}")
            # Pad with last frame if needed
            while len(frames) < self.sample_n_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (self.sample_size, self.sample_size)))
            # Truncate if too many
            frames = frames[:self.sample_n_frames]

        return frames

    def get_video_info(self, idx):
        """
        Get video information for debugging
        """
        video_info = self.video_data[idx]
        videoid = video_info['videoid']
        prompt = self.prompt_mapping.get(videoid, "")

        return {
            'videoid': videoid,
            'prompt': prompt,
            'sampled_indices': video_info['sampled_indices'],
            'sample_count': len([v for v in self.video_data if v['videoid'] == videoid])
        }

    def get_unique_videos(self):
        """
        Get list of unique video IDs and their sample counts
        """
        video_counts = {}
        for video_info in self.video_data:
            videoid = video_info['videoid']
            video_counts[videoid] = video_counts.get(videoid, 0) + 1
        return video_counts

    def test_save_first_ten_videos(self, output_dir="/workspace/video_ctrl/test_output"):
        """
        Test function to save the first 10 video samples as images
        """
        os.makedirs(output_dir, exist_ok=True)

        for i in range(min(10, len(self))):
            try:
                # Get sample without transforms to save original images
                video_info = self.video_data[i]
                videoid = video_info['videoid']
                sampled_indices = video_info['sampled_indices']

                video_path = os.path.join(self.video_folder, videoid)
                frames = self._extract_frames_from_video(video_path, sampled_indices)

                # Get prompt
                prompt = self.prompt_mapping.get(videoid, "No prompt")

                # Create folder for this sample
                sample_dir = os.path.join(output_dir, f"sample_{i:02d}_{videoid.split('.')[0]}")
                os.makedirs(sample_dir, exist_ok=True)

                # Save info
                with open(os.path.join(sample_dir, "info.txt"), "w") as f:
                    f.write(f"Video ID: {videoid}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Sampled indices: {sampled_indices}\n")
                    f.write(f"Number of frames: {len(frames)}\n")

                # Save frames
                for j, frame in enumerate(frames):
                    frame_path = os.path.join(sample_dir, f"frame_{j:02d}_idx_{sampled_indices[j]}.jpg")
                    frame.save(frame_path)

                print(f"Saved sample {i}: {videoid} -> {sample_dir}")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")

        print(f"Test completed! Check {output_dir} directory for results.")


if __name__ == "__main__":
    # 测试数据集
    dataset = PaintDataset(
        video_json_path="/workspace/video_ctrl/video_16_enhance.jsonl",
        video_folder="/workspace/video_ctrl/videos",
        csv_path="/workspace/final_frame/annotation_ctrl.csv",
        sample_size=256,
        sample_n_frames=16
    )

    # 保存前10个样本
    dataset.test_save_first_ten_videos()





