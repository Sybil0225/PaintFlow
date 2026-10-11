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
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(base_savedir)

    config = OmegaConf.load(args.config)

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
    vae = vae.cuda(0)

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
        unet = unet.cuda(0)

        # Load motion modules
        motion_module_paths = sorted(glob.glob(os.path.join(motion_modules_dir, "paintflow.ckpt")))
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
            ).to("cuda:0")

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
                save_videos_grid(sample, f"{savedir}/{sample_idx}-{prompt_text}.gif")
                print(f"Saved to {savedir}/{sample_idx}-{prompt_text}.gif")

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
                        default=".",
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

    args = parser.parse_args()
    main(args)