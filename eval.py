import argparse
import datetime
import os
import json
import glob
from tqdm.auto import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from safetensors.torch import load_file


import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.trainer_utils import is_main_process

# 复用原始代码组件
from paintflow.data.dataset import PaintVideo
from paintflow.models.unet import UNet3DConditionModel
from paintflow.pipelines.pipeline_animation import AnimationPipeline
from paintflow.utils.util import load_weights, save_videos_grid
import cv2


class VideoEvaluator:
    def __init__(self, args):
        self.device = "cuda"
        self.args = args
        self._init_metrics()
        self._init_clip_model()

        # 初始化数据集
        self.test_dataset = PaintVideo(
            csv_path=args.test_csv,
            video_folder=args.test_dir,
            sample_size=args.sample_size,
            sample_stride=args.sample_stride,
            sample_n_frames=args.sample_n_frames,
            is_image=False
        )

        # 初始化DataLoader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            collate_fn=self._collate_fn,
            persistent_workers=False,  # 避免保持工作进程
            prefetch_factor=2  # 减少预取数量
        )

        # 验证数据形状
        self._validate_data_shapes()

        # 创建输出目录
        self.output_dir = f"eval_results/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载基础模型组件
        self._load_base_models()

    def _init_metrics(self):
        device = self.device
        """初始化所有评估指标"""
        # PSNR (范围[0,1])
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        # SSIM (范围[0,1])
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # LPIPS (范围[-1,1])

        self.lpips = lpips.LPIPS(
            net='alex',
            model_path="/workspace/AnimateDiff/models/alex.pth",
            verbose=False  # 关闭冗余输出
        )
        self.lpips.to(device).eval()


    def _collate_fn(self, batch):
        """处理不同长度的视频"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        texts = [item['text'] for item in batch]
        return {
            'pixel_values': pixel_values,
            'text': texts
        }

    def _validate_data_shapes(self):
        """验证数据形状"""
        try:
            sample = next(iter(self.test_loader))
            pixel_values = sample['pixel_values']
            print(f"Debug - sample shape: {pixel_values.shape}")

            assert pixel_values.dim() == 5, "数据应为5维 [B,T,C,H,W]"
            assert pixel_values.shape[2] == 3, "需要RGB三通道"
            assert pixel_values.shape[3] == self.args.sample_size, "高度不匹配"
            assert pixel_values.shape[4] == self.args.sample_size, "宽度不匹配"

        except Exception as e:
            print(f"数据验证失败: {str(e)}")
            # 打印第一个样本的详细信息
            debug_sample = self.test_dataset[0]
            print(f"第一个样本类型: {type(debug_sample)}")
            if isinstance(debug_sample, dict):
                print(f"包含的键: {debug_sample.keys()}")
                if 'pixel_values' in debug_sample:
                    print(f"pixel_values形状: {debug_sample['pixel_values'].shape}")
            raise

    def _load_vae(self):
        """加载VAE权重"""
        if self.args.vae_path:
            print(f"Loading custom VAE from: {self.args.vae_path}")
            try:
                if self.args.vae_path.endswith(".safetensors"):
                    state_dict = load_file(self.args.vae_path, device="cpu")
                    state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}
                    vae = AutoencoderKL.from_pretrained(
                        self.args.pretrained_model_path,
                        subfolder="vae"
                    )
                    vae.load_state_dict(state_dict, strict=False)
                else:
                    vae = AutoencoderKL.from_pretrained(self.args.vae_path)
            except Exception as e:
                print(f"Failed to load custom VAE: {str(e)}")
                print("Falling back to default VAE")
                vae = AutoencoderKL.from_pretrained(
                    self.args.pretrained_model_path,
                    subfolder="vae"
                )
        else:
            print("Using default VAE from pretrained model")
            vae = AutoencoderKL.from_pretrained(
                self.args.pretrained_model_path,
                subfolder="vae"
            )
        return vae.to(self.device)

    def _load_unet(self):
        """加载UNet权重"""
        inference_config = OmegaConf.load(self.args.inference_config)

        if self.args.unet_path:
            print(f"Loading custom UNet from: {self.args.unet_path}")
            try:
                if self.args.unet_path.endswith(".safetensors"):
                    state_dict = load_file(self.args.unet_path, device="cpu")
                else:
                    state_dict = torch.load(self.args.unet_path, map_location="cpu")

                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                unet = UNet3DConditionModel.from_pretrained_2d(
                    self.args.pretrained_model_path,
                    subfolder="unet",
                    unet_additional_kwargs=OmegaConf.to_container(
                        inference_config.unet_additional_kwargs
                    )
                )

                missing, unexpected = unet.load_state_dict(state_dict, strict=False)
                print(f"UNet missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"Failed to load custom UNet: {str(e)}")
                print("Falling back to default UNet")
                unet = UNet3DConditionModel.from_pretrained_2d(
                    self.args.pretrained_model_path,
                    subfolder="unet",
                    unet_additional_kwargs=OmegaConf.to_container(
                        inference_config.unet_additional_kwargs
                    )
                )
        else:
            print("Using default UNet from pretrained model")
            unet = UNet3DConditionModel.from_pretrained_2d(
                self.args.pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(
                    inference_config.unet_additional_kwargs
                )
            )
        return unet.to(self.device)

    def _load_base_models(self):
        """加载共享模型组件"""
        # 文本编码器
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_path,
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_path,
            subfolder="text_encoder"
        ).to(self.device)

        # VAE
        self.vae = self._load_vae()

        # UNet
        self.unet = self._load_unet()

    def _prepare_pipeline(self, motion_module_path):
        """准备带权重的pipeline"""
        pipeline = AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(
                OmegaConf.load(self.args.inference_config).noise_scheduler_kwargs
            )),
        ).to(self.device)

        return load_weights(
            pipeline,
            motion_module_path=motion_module_path,
            motion_module_lora_configs=[],
            adapter_lora_path="",
            adapter_lora_scale=1.0
        )

    def _init_clip_model(self):
        """初始化CLIP模型用于文本-视频相似度计算"""
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _compute_clip_score(self, video_frames: torch.Tensor, text: str) -> float:
        """
        修改为无界输出的CLIP Score计算（类似torchmetrics实现）
        返回范围：理论上无界（通常0-100+）
        """
        # 1. 输入验证和转换
        assert video_frames.dim() == 4, f"视频帧应为[T,C,H,W]，实际得到{video_frames.shape}"
        T, C, H, W = video_frames.shape

        # 转换为0-255范围的numpy数组
        if video_frames.min() >= -1 and video_frames.max() <= 1:  # [-1,1]范围
            frames = [(frame.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                      for frame in video_frames]
        else:  # 假设[0,1]范围
            frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                      for frame in video_frames]

        # 2. 文本处理
        text = str(text).strip()
        if not text:
            return 0.0

        # 3. 使用CLIP模型计算原始相似度
        with torch.no_grad():
            # 提取图像特征
            image_inputs = self.clip_processor(
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            image_features = self.clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # 提取文本特征
            text_inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # 计算余弦相似度（无界输出）
            raw_scores = (image_features @ text_features.T) * 100  # 放大100倍（标准CLIP做法）
            final_score = raw_scores.mean().item()

        # print(f"Debug: 原始CLIP分数={raw_scores.cpu().numpy().round(2)}, 平均分数={final_score:.2f}")
        return final_score

    def _compute_metrics(self, generated, real, prompt=None):
        device = self.device
        generated = generated.to(device)
        real = real.to(device)
        """处理多维度视频输入的指标计算"""
        # 统一调整为 [B, C, T, H, W] 格式
        if generated.dim() == 5:
            generated = generated.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]
        if real.dim() == 5:
            real = real.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        # 展平时间维度 [B, C, T, H, W] -> [B*T, C, H, W]
        b, c, t, h, w = generated.shape
        generated_flat = generated.reshape(b * t, c, h, w)
        real_flat = real.reshape(b * t, c, h, w)

        # 转换数值范围
        generated_01 = ((generated_flat + 1) / 2).to(device)  # [-1,1] -> [0,1]
        real_01 = ((real_flat + 1) / 2).to(device)

        metrics = {
            "psnr": self.psnr(generated_01, real_01).cpu().item(),
            "ssim": self.ssim(generated_01, real_01).cpu().item(),
            "lpips": self.lpips(generated_flat, real_flat).mean().cpu().item()
        }

        # 新增CLIP Score计算
        if prompt is not None:
            # 取batch中第一个视频计算（避免重复计算）
            sample_video = generated[0].permute(1, 0, 2, 3)  # [C,T,H,W] -> [T,C,H,W]
            metrics["clip_score"] = self._compute_clip_score(sample_video, prompt)

        return metrics

    def evaluate(self):
        print("1. 开始评估流程")
        results = {}

        motion_modules = sorted(glob.glob(os.path.join(self.args.motion_modules_dir, "*.ckpt")))
        print(f"2. 找到 {len(motion_modules)} 个motion modules")

        for mm_path in motion_modules:
            mm_name = Path(mm_path).stem
            print(f"3. 正在评估: {mm_name}")

            pipeline = self._prepare_pipeline(mm_path)
            print("4. Pipeline初始化完成")

            mm_results = {"samples": [], "average": {}}

            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Processing batches")):
                print(f"\n5. 处理第 {batch_idx} 个batch")
                real_videos = batch["pixel_values"].to(self.device)
                prompts = batch["text"]
                print(f"6. 真实视频形状: {real_videos.shape}, 提示词数量: {len(prompts)}")

                with torch.cuda.amp.autocast():
                    generated_videos = []
                    for prompt in prompts:
                        print(f"7. 生成视频: {prompt[:50]}...")  # 打印前50字符避免刷屏
                        output = pipeline(
                            prompt,
                            num_inference_steps=self.args.num_steps,
                            guidance_scale=self.args.guidance_scale,
                            video_length=self.args.sample_n_frames,
                            height=self.args.sample_size,
                            width=self.args.sample_size
                        ).videos
                        generated_videos.append(output)
                    generated_videos = torch.cat(generated_videos)
                    generated_videos = generated_videos.permute(0, 2, 1, 3, 4)


                # 计算指标
                batch_metrics = self._compute_metrics(generated_videos, real_videos, prompt=prompts[0] if prompts else None)

                # 保存样本结果
                for i in range(len(prompts)):
                    sample_data = {
                        "prompt": prompts[i],
                        "metrics": batch_metrics  # 直接使用整个batch的指标
                    }
                    mm_results["samples"].append(sample_data)

                # 保存生成视频
                if self.args.save_samples:
                    save_path = os.path.join(self.output_dir, mm_name)
                    os.makedirs(save_path, exist_ok=True)
                    save_videos_grid(generated_videos,
                                     os.path.join(save_path, f"batch_{len(mm_results['samples'])}.mp4"))

            # 计算平均指标
            avg_metrics = {
                "psnr": float(np.mean([s["metrics"]["psnr"] for s in mm_results["samples"]])),
                "ssim": float(np.mean([s["metrics"]["ssim"] for s in mm_results["samples"]])),
                "lpips": float(np.mean([s["metrics"]["lpips"] for s in mm_results["samples"]])),
                "clip_score": float(np.mean([s["metrics"]["clip_score"] for s in mm_results["samples"]]))
            }
            mm_results["average"] = avg_metrics

            results[mm_name] = mm_results
            torch.cuda.empty_cache()
        # 保存最终结果
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument("--test_csv", type=str, default="/workspace/test/test.csv", help="测试集CSV路径")
    parser.add_argument("--test_dir", type=str, default="/workspace/test/videos", help="测试视频目录")
    parser.add_argument("--sample_size", type=int, default=256, help="视频分辨率")
    parser.add_argument("--sample_stride", type=int, default=1, help="帧采样步长")
    parser.add_argument("--sample_n_frames", type=int, default=16, help="每段视频采样帧数")

    # 模型参数
    parser.add_argument("--pretrained_model_path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion")
    parser.add_argument("--vae-path", type=str, default="/workspace/AnimateDiff/SD_1_5/vae/vae_final.safetensors",
                        help="Path to custom VAE weights")
    parser.add_argument("--unet-path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion/unet/diffusion_pytorch_model.safetensors",
                        help="Path to custom UNet weights")
    parser.add_argument("--motion_modules_dir", type=str, default="/workspace/AnimateDiff/PaintFlow/outputs/training-2025-07-09T10-53-55/checkpoints")
    parser.add_argument("--inference_config", type=str, default="/workspace/AnimateDiff/PaintFlow/configs/inference/inference-v5.yaml")


    # 生成参数
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=4)

    # 其他参数
    parser.add_argument("--save_samples", action="store_true", help="是否保存生成样本")

    args = parser.parse_args()

    evaluator = VideoEvaluator(args)
    results = evaluator.evaluate()
    print("\nEvaluation complete. Results saved to:", evaluator.output_dir)