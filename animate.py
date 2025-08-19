import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler


from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from paintflow.models.unet import UNet3DConditionModel
from paintflow.models.sparse_controlnet import SparseControlNetModel
from paintflow.pipelines.pipeline_animation import AnimationPipeline
from paintflow.utils.util import save_videos_grid
from paintflow.utils.util import load_weights, auto_download
from paintflow.data.sketch_extractor import SketchExtractor
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(base_savedir)

    config = OmegaConf.load(args.config)
    sketch_extractor = SketchExtractor() if args.enable_sketch_sequence else None
    # motion modules directory from CLI argument
    motion_modules_dir = args.motion_modules_dir
    if not os.path.exists(motion_modules_dir):
        raise ValueError(f"motion_modules_dir '{motion_modules_dir}' does not exist.")

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda(1)

    # Load VAE
    if args.vae_path:
        print(f"Loading custom VAE from: {args.vae_path}")
        try:
            if args.vae_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(args.vae_path, device="cpu")
                state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}
                vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
                vae.load_state_dict(state_dict, strict=False)
            else:
                vae = AutoencoderKL.from_pretrained(args.vae_path)
        except Exception as e:
            print(f"Failed to load custom VAE: {str(e)}")
            print("Falling back to default VAE")
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    else:
        print("Using default VAE from pretrained model")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    vae = vae.cuda(1)

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

        # Load UNet
        if args.unet_path:
            print(f"Loading custom UNet from: {args.unet_path}")
            try:
                if args.unet_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(args.unet_path, device="cpu")
                else:
                    state_dict = torch.load(args.unet_path, map_location="cpu")

                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                unet = UNet3DConditionModel.from_pretrained_2d(
                    args.pretrained_model_path,
                    subfolder="unet",
                    unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
                )

                missing_keys, unexpected_keys = unet.load_state_dict(state_dict, strict=False)
                print(f"UNet missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    print(f"First few unexpected keys: {unexpected_keys[:5]}...")
            except Exception as e:
                print(f"Failed to load custom UNet: {str(e)}")
                print("Falling back to default UNet")
                unet = UNet3DConditionModel.from_pretrained_2d(
                    args.pretrained_model_path,
                    subfolder="unet",
                    unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
                )
        else:
            print("Using default UNet from pretrained model")
            unet = UNet3DConditionModel.from_pretrained_2d(
                args.pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
            )
        unet = unet.cuda(1)

        # Load motion modules
        motion_module_paths = sorted(glob.glob(os.path.join(motion_modules_dir, "*.ckpt")))
        if not motion_module_paths:
            raise ValueError(f"No motion module checkpoints found in '{motion_modules_dir}'")

        for motion_module_path in motion_module_paths:
            motion_module_name = Path(motion_module_path).stem
            savedir = os.path.join(base_savedir, motion_module_name)
            os.makedirs(savedir, exist_ok=True)

            print(f"Using motion module: {motion_module_path}")

            pipeline = AnimationPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=DDIMScheduler
    (**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda:1")

            pipeline = load_weights(
                pipeline,
                motion_module_path=motion_module_path,
                motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
                adapter_lora_path=model_config.get("adapter_lora_path", ""),
                adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
                dreambooth_model_path=model_config.get("dreambooth_path", ""),
                lora_model_path=model_config.get("lora_model_path", ""),
                lora_alpha=model_config.get("lora_alpha", 0.8),
            ).to("cuda:0")

            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(
                model_config.n_prompt) == 1 else model_config.n_prompt
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

            config[model_idx].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[model_idx].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} with {motion_module_path}...")

                sample = pipeline(
                    prompt,
                    negative_prompt=n_prompt,
                    num_inference_steps=model_config.steps,
                    guidance_scale=model_config.guidance_scale,
                    width=model_config.W,
                    height=model_config.H,
                    video_length=model_config.L,
                ).videos

                prompt_text = "-".join((prompt.replace("/", "").split(" ")[:10]))
                original_video_path = f"{savedir}/{sample_idx}-{prompt_text}.gif"

                # 如果启用线稿序列功能
                if args.enable_sketch_sequence:
                    try:
                        # 从sample中提取帧（sample是torch tensor格式）
                        # sample 的形状应该是 [1, C, T, H, W] 或 [1, T, C, H, W]
                        sample_np = sample[0].cpu().numpy()

                        print(f"调试: sample shape: {sample_np.shape}")

                        # 确保数据类型正确
                        if sample_np.dtype != np.float32:
                            sample_np = sample_np.astype(np.float32)

                        # 将数值范围从 [-1, 1] 转换到 [0, 255]
                        sample_np = (sample_np + 1.0) * 127.5
                        sample_np = np.clip(sample_np, 0, 255).astype(np.uint8)

                        if len(sample_np.shape) == 4:
                            if sample_np.shape[0] == 3 and sample_np.shape[1] > 3:  # [C=3, T, H, W]
                                sample_np = sample_np.transpose(1, 2, 3, 0)  # [T, H, W, C]
                                print(f"调试: 转换后 shape (C,T,H,W->T,H,W,C): {sample_np.shape}")
                            elif sample_np.shape[1] == 3:  # [T, C=3, H, W]
                                sample_np = sample_np.transpose(0, 2, 3, 1)  # [T, H, W, C]
                                print(f"调试: 转换后 shape (T,C,H,W->T,H,W,C): {sample_np.shape}")
                            else:
                                print(f"警告: 无法识别的维度格式: {sample_np.shape}")
                                # 尝试猜测最可能的格式
                                if sample_np.shape[-1] == 3:  # 最后一维是通道
                                    pass  # 已经是正确格式
                                elif sample_np.shape[0] == model_config.L:  # 第一维是时间
                                    # 可能是 [T, H, W, C] 但 C 不是 3
                                    if sample_np.shape[-1] != 3:
                                        print(f"错误: 最后一维不是3个通道: {sample_np.shape}")
                                        continue
                                else:
                                    print(f"错误: 无法处理的维度: {sample_np.shape}")
                                    continue

                        frames = []
                        for t in range(sample_np.shape[0]):
                            frame = sample_np[t]  # [H, W, C]
                            print(f"调试: 帧 {t} shape: {frame.shape}")

                            # 确保是RGB格式
                            if len(frame.shape) == 3 and frame.shape[-1] == 3:
                                frames.append(frame)
                            elif len(frame.shape) == 2:  # 灰度图
                                # 转换为RGB
                                frame_rgb = np.stack([frame, frame, frame], axis=-1)
                                frames.append(frame_rgb)
                                print(f"调试: 将灰度图转换为RGB")
                            else:
                                print(f"警告: 跳过帧 {t}，形状不正确: {frame.shape}")
                                continue

                        if not frames:
                            raise ValueError("没有有效的帧可以处理")

                        print(f"调试: 成功处理了 {len(frames)} 帧")

                        # 获取第一帧（用于线稿生成）
                        first_frame = frames[0]
                        first_frame_pil = Image.fromarray(first_frame)

                        # 使用LoRA风格生成线稿，然后创建4帧渐进序列
                        print("正在使用LoRA风格生成线稿序列...")
                        sketch_frames = sketch_extractor.generate_lora_sketch_sequence(
                            first_frame_pil,
                            prompt=prompt,
                            num_frames=2,  # 2帧
                            pipeline=pipeline,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            width=model_config.W,
                            height=model_config.H
                        )

                        # 合并生成18帧视频 (2帧线稿 + 16帧原始)
                        enhanced_video_path = f"{savedir}/{sample_idx:03d}-{prompt_idx:02d}-{prompt_text}_18frames.gif"
                        sketch_extractor.combine_frames_to_video_only(
                            sketch_frames,
                            frames,
                            enhanced_video_path,
                            fps=args.output_fps
                        )

                        print(f"Saved enhanced 18-frame video to {enhanced_video_path}")

                    except Exception as e:
                        print(f"线稿序列生成失败: {e}")
                        print("继续使用原始视频...")
                        import traceback
                        traceback.print_exc()

            sample_idx += 1

            OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion/")
    parser.add_argument("--inference-config", type=str,
                        default="/workspace/AnimateDiff/PaintFlow/configs/inference/inference-v7.yaml")
    parser.add_argument("--config", type=str,
                        default="/workspace/AnimateDiff/PaintFlow/configs/prompts/1_animate/animate.yaml")

    parser.add_argument("--motion_modules_dir", type=str,
                        default="/workspace/AnimateDiff/PaintFlow/outputs/PaintEnh_stage128_prog5_progAttn_h6b1_lr2e6_bs3/checkpoints",
                        help="Path to the directory containing motion module checkpoints")

    # Add new arguments for custom UNet and VAE paths
    parser.add_argument("--unet-path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion/unet/diffusion_pytorch_model.safetensors",
                        help="Path to custom UNet weights (either .ckpt or .safetensors file)")
    parser.add_argument("--vae-path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion/vae/diffusion_pytorch_model.safetensors",
                        help="Path to custom VAE weights (either directory, .ckpt or .safetensors file)")

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)

    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--enable-sketch-sequence", action="store_true", default=True,
                        help="Enable sketch sequence generation")
    parser.add_argument("--sketch-method", type=str, default="canny",
                        choices=["canny", "sobel", "laplacian", "threshold"],
                        help="Method for line art extraction")
    parser.add_argument("--sketch-frames", type=int, default=2,
                        help="Number of sketch frames to generate")
    parser.add_argument("--output-fps", type=float, default=8.0,
                        help="Output video frame rate")

    args = parser.parse_args()
    main(args)