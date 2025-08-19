# animate_gui.py
# 运行方式：python animate_gui.py
# 访问 http://127.0.0.1:7860

import os
import io
import gc
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import gradio as gr
import numpy as np
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from paintflow.models.unet import UNet3DConditionModel
from paintflow.models.sparse_controlnet import SparseControlNetModel
from paintflow.pipelines.pipeline_animation import AnimationPipeline
from paintflow.utils.util import load_weights

# ========== 全局缓存（避免重复加载） ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

class Pipelines:
    text = None
    ctrl = None
    text_components = {}
    ctrl_components = {}

def _to_uint8_frames(video_tensor: torch.Tensor) -> List[Image.Image]:
    """
    将返回的 videos tensor(B, C, T, H, W) -> List[PIL.Image]，每帧为RGB
    """
    if video_tensor.dim() != 5:
        raise ValueError(f"expect videos with B,C,T,H,W, got shape {tuple(video_tensor.shape)}")

    # 只取 batch 中第一个视频
    vid = video_tensor[0]  # (C, T, H, W)
    if vid.shape[0] == 3:
        vid = vid.permute(1, 2, 3, 0).contiguous()  # (T, H, W, C)
    else:
        raise ValueError(f"expect 3 channels, got {vid.shape[0]}")

    # 将 [-1,1] 或 [0,1] 归一到 [0,255]
    v = vid.detach().cpu().float().numpy()
    if v.min() < 0.0:
        v = (v + 1.0) / 2.0
    v = np.clip(v, 0.0, 1.0)
    v = (v * 255.0).astype(np.uint8)

    frames = [Image.fromarray(v[t]).convert("RGB") for t in range(v.shape[0])]
    return frames

def _frames_to_gif_bytes(frames: List[Image.Image], fps: float = 8.0) -> bytes:
    """
    将帧列表写成 GIF 并返回字节流
    """
    if not frames:
        raise ValueError("Empty frames")
    duration = int(1000.0 / max(fps, 1e-6))  # ms
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    buf.seek(0)
    return buf.read()

def _save_gif_to_temp(gif_bytes: bytes) -> str:
    """Persist GIF bytes to a temp file and return its filepath for Gradio."""
    os.makedirs("/tmp/anim_gifs", exist_ok=True)
    path = f"/tmp/anim_gifs/{int(time.time() * 1000)}.gif"
    with open(path, "wb") as f:
        f.write(gif_bytes)
    return path

def _save_frames_to_temp(frames: List[Image.Image]) -> List[str]:
    """Save frames as PNG files and return list of file paths (avoids numpy.bool8 issues)."""
    os.makedirs("/tmp/anim_frames", exist_ok=True)
    ts = int(time.time() * 1000)
    paths = []
    for i, fr in enumerate(frames):
        p = f"/tmp/anim_frames/{ts}_{i:03d}.png"
        # ensure RGB and uint8
        fr.convert("RGB").save(p, format="PNG")
        paths.append(p)
    return paths

def _resize_pil(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)

# ========== 文生视频 ==========

def init_text_pipeline(
    pretrained_model_path: str,
    unet_path: Optional[str],
    vae_path: Optional[str],
    motion_module_path: Optional[str],
    dreambooth_path: Optional[str],
    lora_path: Optional[str],
    lora_alpha: float,
    adapter_lora_path: Optional[str],
    adapter_lora_scale: float,
    use_xformers: bool,
):
    # tokenizer, text_encoder, vae, unet, scheduler, pipeline
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)

    # VAE
    if vae_path and Path(vae_path).exists():
        try:
            if vae_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(vae_path, device="cpu")
                state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}
                vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
                vae.load_state_dict(state_dict, strict=False)
            else:
                vae = AutoencoderKL.from_pretrained(vae_path)
        except Exception as e:
            print(f"[WARN] load custom VAE failed: {e}, fallback to default.")
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    vae = vae.to(device)

    # UNet
    if unet_path and Path(unet_path).exists():
        try:
            if unet_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(unet_path, device="cpu")
            else:
                state_dict = torch.load(unet_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            unet = UNet3DConditionModel.from_pretrained_2d(
                pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs={}
            )
            unet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[WARN] load custom UNet failed: {e}, fallback to default.")
            unet = UNet3DConditionModel.from_pretrained_2d(
                pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs={}
            )
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs={}
        )
    unet = unet.to(device)

    # scheduler 用默认 DDIM
    scheduler = DDIMScheduler()

    pipe = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    # 加载运动模块 / DreamBooth / LoRA / Adapter LoRA
    pipe = load_weights(
        pipe,
        motion_module_path=motion_module_path or "",
        motion_module_lora_configs=[],
        adapter_lora_path=adapter_lora_path or "",
        adapter_lora_scale=float(adapter_lora_scale),
        dreambooth_model_path=dreambooth_path or "",
        lora_model_path=lora_path or "",
        lora_alpha=float(lora_alpha),
    ).to(device)

    if use_xformers:
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] enable xformers failed: {e}")

    Pipelines.text = pipe
    Pipelines.text_components = dict(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
    )

def generate_text_video(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    video_length: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    fps: float,
    pretrained_model_path: str,
    unet_path: str,
    vae_path: str,
    motion_module_path: str,
    dreambooth_path: str,
    lora_path: str,
    lora_alpha: float,
    adapter_lora_path: str,
    adapter_lora_scale: float,
    use_xformers: bool,
):
    if Pipelines.text is None:
        init_text_pipeline(
            pretrained_model_path=pretrained_model_path,
            unet_path=unet_path or None,
            vae_path=vae_path or None,
            motion_module_path=motion_module_path or None,
            dreambooth_path=dreambooth_path or None,
            lora_path=lora_path or None,
            lora_alpha=float(lora_alpha),
            adapter_lora_path=adapter_lora_path or None,
            adapter_lora_scale=float(adapter_lora_scale),
            use_xformers=use_xformers,
        )

    pipe = Pipelines.text

    # 随机种子
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
    else:
        torch.seed()

    with torch.no_grad():
        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            video_length=int(video_length),
        ).videos

    frames = _to_uint8_frames(videos)
    # 适配 256x256 展示
    frames = [_resize_pil(f, (width, height)) for f in frames]
    gif_bytes = _frames_to_gif_bytes(frames, fps=fps)
    gif_path = _save_gif_to_temp(gif_bytes)
    frame_paths = _save_frames_to_temp(frames)
    return gif_path, frame_paths

# ========== 图生视频（带 ControlNet） ==========

def init_ctrl_pipeline(
    pretrained_model_path: str,
    controlnet_ckpt_path: str,
    controlnet_config_path_unused: str = "",
    unet_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    dreambooth_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    lora_alpha: float = 0.8,
    adapter_lora_path: Optional[str] = None,
    adapter_lora_scale: float = 1.0,
    use_xformers: bool = True,
):
    # 基础组件
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)

    # VAE
    if vae_path and Path(vae_path).exists():
        try:
            if vae_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(vae_path, device="cpu")
                state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}
                vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
                vae.load_state_dict(state_dict, strict=False)
            else:
                vae = AutoencoderKL.from_pretrained(vae_path)
        except Exception as e:
            print(f"[WARN] load custom VAE failed: {e}, fallback to default.")
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    vae = vae.to(device)

    # UNet
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs={}
    )
    if unet_path and Path(unet_path).exists():
        try:
            if unet_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(unet_path, device="cpu")
            else:
                state_dict = torch.load(unet_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            unet.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[WARN] load custom UNet failed: {e}")
    unet = unet.to(device)

    # ControlNet
    controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs={})
    if not controlnet_ckpt_path or not Path(controlnet_ckpt_path).exists():
        raise ValueError("Please provide a valid ControlNet checkpoint path.")
    state_dict = torch.load(controlnet_ckpt_path, map_location="cpu")
    if "controlnet" in state_dict:
        state_dict = state_dict["controlnet"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # 去掉可能不兼容的键
    state_dict = {k: v for k, v in state_dict.items() if "pos_encoder.pe" not in k}
    if "animatediff_config" in state_dict:
        state_dict.pop("animatediff_config")
    controlnet.load_state_dict(state_dict, strict=False)
    controlnet = controlnet.to(device)

    scheduler = DDIMScheduler()

    pipe = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    ).to(device)

    # 支持 DreamBooth / LoRA / Adapter LoRA
    pipe = load_weights(
        pipe,
        motion_module_path="",  # 可按需添加
        motion_module_lora_configs=[],
        adapter_lora_path=adapter_lora_path or "",
        adapter_lora_scale=float(adapter_lora_scale),
        dreambooth_model_path=dreambooth_path or "",
        lora_model_path=lora_path or "",
        lora_alpha=float(lora_alpha),
    ).to(device)

    if use_xformers:
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
            pipe.controlnet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] enable xformers failed: {e}")

    Pipelines.ctrl = pipe
    Pipelines.ctrl_components = dict(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )

def _image_to_control_tensor(img: Image.Image, W: int, H: int, L: int, vae: Optional[AutoencoderKL] = None) -> torch.Tensor:
    """
    将上传的图片转为 ControlNet 需要的形状 (1, 3, L, H, W)，简单重复 L 帧
    """
    img = img.convert("RGB").resize((W, H), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    arr = arr.unsqueeze(0)  # (1, 3, H, W)
    arr = arr.repeat(L, 1, 1, 1)  # (L, 3, H, W)
    arr = arr.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # (1, 3, L, H, W)
    return arr.to(device)

def generate_ctrl_video(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    video_length: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    fps: float,
    pretrained_model_path: str,
    unet_path: str,
    vae_path: str,
    controlnet_ckpt_path: str,
    control_image: Image.Image,
    dreambooth_path: str,
    lora_path: str,
    lora_alpha: float,
    adapter_lora_path: str,
    adapter_lora_scale: float,
    use_xformers: bool,
):
    if Pipelines.ctrl is None:
        init_ctrl_pipeline(
            pretrained_model_path=pretrained_model_path,
            controlnet_ckpt_path=controlnet_ckpt_path,
            unet_path=unet_path or None,
            vae_path=vae_path or None,
            dreambooth_path=dreambooth_path or None,
            lora_path=lora_path or None,
            lora_alpha=float(lora_alpha),
            adapter_lora_path=adapter_lora_path or None,
            adapter_lora_scale=float(adapter_lora_scale),
            use_xformers=use_xformers,
        )

    if control_image is None:
        raise gr.Error("Please upload a control image first.")

    pipe = Pipelines.ctrl

    # 随机种子
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
    else:
        torch.seed()

    controlnet_images = _image_to_control_tensor(control_image, width, height, video_length, Pipelines.ctrl_components.get("vae"))

    with torch.no_grad():
        videos = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            video_length=int(video_length),
            controlnet_images=controlnet_images,
            controlnet_image_index=list(range(video_length)),
        ).videos

    frames = _to_uint8_frames(videos)
    frames = [_resize_pil(f, (width, height)) for f in frames]
    gif_bytes = _frames_to_gif_bytes(frames, fps=fps)
    gif_path = _save_gif_to_temp(gif_bytes)
    frame_paths = _save_frames_to_temp(frames)
    return gif_path, frame_paths

# ========== Gradio UI ==========

CSS = """
.wrap-256 .preview-image img { max-width: 280px !important; max-height: 280px !important; }
.wrap-256 .thumbnail-item img { width: 128px !important; height: 128px !important; object-fit: contain; }
"""

# 移除主题，避免紫色主题
with gr.Blocks(css=CSS) as demo:
    gr.Markdown("## Animation Generation (Text-to-Video / Image-to-Video) - Optimized for 256x256")

    with gr.Tabs():
        # 文生视频
        with gr.Tab("Text-to-Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_t = gr.Textbox(label="Prompt", value="a cute cat dancing in the snow, 8-bit pixel art, loop")
                    negative_t = gr.Textbox(label="Negative Prompt", value="")
                    width_t = gr.Slider(64, 512, value=256, step=16, label="Width")
                    height_t = gr.Slider(64, 512, value=256, step=16, label="Height")
                    length_t = gr.Slider(4, 64, value=16, step=1, label="Video Length (frames)")
                    steps_t = gr.Slider(10, 100, value=30, step=1, label="Sampling Steps")
                    scale_t = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                    fps_t = gr.Slider(1.0, 24.0, value=8.0, step=1.0, label="FPS")
                    seed_t = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")

                    gr.Markdown("Model / Weights (leave blank to use defaults)")
                    pretrained_t = gr.Textbox(label="Pretrained Model Path", value="/workspace/AD-Evo-Tuner/models/StableDiffusion")
                    unet_t = gr.Textbox(label="UNet Weights Path (.ckpt/.safetensors, optional)", value="")
                    vae_t = gr.Textbox(label="VAE Weights Path (.ckpt/.safetensors/folder, optional)", value="")
                    motion_t = gr.Textbox(label="Motion Module Path (.ckpt, optional)", value="")
                    dreambooth_t = gr.Textbox(label="DreamBooth Model Path (optional)", value="")
                    lora_t = gr.Textbox(label="LoRA Weights Path (optional)", value="")
                    lora_alpha_t = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="LoRA Alpha")
                    adapter_lora_t = gr.Textbox(label="Adapter LoRA Path (optional)", value="")
                    adapter_scale_t = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Adapter LoRA Scale")
                    xformers_t = gr.Checkbox(label="Enable xFormers (if available)", value=True)

                    run_t = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1, elem_classes=["wrap-256"]):
                    # 使用 filepath 避免 numpy.bool8 相关问题，同时支持 GIF 动画展示
                    out_gif_t = gr.Image(label="GIF Preview", type="filepath", interactive=False)
                    out_gallery_t = gr.Gallery(label="Frames (click to enlarge)", columns=4, height=320)

            def on_run_text(
                prompt, negative, w, h, l, steps, scale, fps, seed,
                pretrained, unet, vae, motion, dreambooth, lora, lora_alpha, adapter_lora, adapter_scale, xformers
            ):
                gif_path, frame_paths = generate_text_video(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=int(w),
                    height=int(h),
                    video_length=int(l),
                    steps=int(steps),
                    guidance_scale=float(scale),
                    seed=int(seed),
                    fps=float(fps),
                    pretrained_model_path=pretrained,
                    unet_path=unet,
                    vae_path=vae,
                    motion_module_path=motion,
                    dreambooth_path=dreambooth,
                    lora_path=lora,
                    lora_alpha=float(lora_alpha),
                    adapter_lora_path=adapter_lora,
                    adapter_lora_scale=float(adapter_scale),
                    use_xformers=bool(xformers),
                )
                return gif_path, frame_paths

            run_t.click(
                fn=on_run_text,
                inputs=[prompt_t, negative_t, width_t, height_t, length_t, steps_t, scale_t, fps_t, seed_t,
                        pretrained_t, unet_t, vae_t, motion_t, dreambooth_t, lora_t, lora_alpha_t, adapter_lora_t, adapter_scale_t, xformers_t],
                outputs=[out_gif_t, out_gallery_t]
            )

        # 图生视频
        with gr.Tab("Image-to-Video (ControlNet)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_c = gr.Textbox(label="Prompt", value="a robot walking forward, line-art guidance")
                    negative_c = gr.Textbox(label="Negative Prompt", value="")
                    control_img = gr.Image(label="Control Image (will be repeated to L frames)", type="pil")
                    width_c = gr.Slider(64, 512, value=256, step=16, label="Width")
                    height_c = gr.Slider(64, 512, value=256, step=16, label="Height")
                    length_c = gr.Slider(4, 64, value=16, step=1, label="Video Length (frames)")
                    steps_c = gr.Slider(10, 100, value=30, step=1, label="Sampling Steps")
                    scale_c = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                    fps_c = gr.Slider(1.0, 24.0, value=8.0, step=1.0, label="FPS")
                    seed_c = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")

                    gr.Markdown("Model / Weights (ControlNet checkpoint required)")
                    pretrained_c = gr.Textbox(label="Pretrained Model Path", value="/workspace/AD-Evo-Tuner/models/StableDiffusion")
                    unet_c = gr.Textbox(label="UNet Weights Path (.ckpt/.safetensors, optional)", value="")
                    vae_c = gr.Textbox(label="VAE Weights Path (.ckpt/.safetensors/folder, optional)", value="")
                    controlnet_ckpt_c = gr.Textbox(label="ControlNet Checkpoint Path (.ckpt)", value="")
                    dreambooth_c = gr.Textbox(label="DreamBooth Model Path (optional)", value="")
                    lora_c = gr.Textbox(label="LoRA Weights Path (optional)", value="")
                    lora_alpha_c = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="LoRA Alpha")
                    adapter_lora_c = gr.Textbox(label="Adapter LoRA Path (optional)", value="")
                    adapter_scale_c = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Adapter LoRA Scale")
                    xformers_c = gr.Checkbox(label="Enable xFormers (if available)", value=True)

                    run_c = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1, elem_classes=["wrap-256"]):
                    out_gif_c = gr.Image(label="GIF Preview", type="filepath", interactive=False)
                    out_gallery_c = gr.Gallery(label="Frames (click to enlarge)", columns=4, height=320)

            def on_run_ctrl(
                prompt, negative, img, w, h, l, steps, scale, fps, seed,
                pretrained, unet, vae, controlnet_ckpt, dreambooth, lora, lora_alpha, adapter_lora, adapter_scale, xformers
            ):
                gif_path, frame_paths = generate_ctrl_video(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=int(w),
                    height=int(h),
                    video_length=int(l),
                    steps=int(steps),
                    guidance_scale=float(scale),
                    seed=int(seed),
                    fps=float(fps),
                    pretrained_model_path=pretrained,
                    unet_path=unet,
                    vae_path=vae,
                    controlnet_ckpt_path=controlnet_ckpt,
                    control_image=img,
                    dreambooth_path=dreambooth,
                    lora_path=lora,
                    lora_alpha=float(lora_alpha),
                    adapter_lora_path=adapter_lora,
                    adapter_lora_scale=float(adapter_scale),
                    use_xformers=bool(xformers),
                )
                return gif_path, frame_paths

            run_c.click(
                fn=on_run_ctrl,
                inputs=[prompt_c, negative_c, control_img, width_c, height_c, length_c, steps_c, scale_c, fps_c, seed_c,
                        pretrained_c, unet_c, vae_c, controlnet_ckpt_c, dreambooth_c, lora_c, lora_alpha_c, adapter_lora_c, adapter_scale_c, xformers_c],
                outputs=[out_gif_c, out_gallery_c]
            )

if __name__ == "__main__":
    # 使用 queue，限制并发，避免长时计算导致前端报错
    demo.queue(concurrency_count=1, max_size=10, status_update_rate=1).launch(
        server_name="0.0.0.0", server_port=7860, inbrowser=False, share=True
    )