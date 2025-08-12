import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

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
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process_sketch_to_scribble(image_path, target_size=(256, 256)):
    """
    Convert a sketch image to black background with white lines (scribble format).
    Same processing as in dataset.py

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size (width, height)

    Returns:
        PIL.Image: Processed scribble image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Invert colors to make black background and white lines
    mean_value = np.mean(gray)
    if mean_value > 127:
        # White background, black lines -> invert to black background, white lines
        black_background_sketch = cv2.bitwise_not(gray)
    else:
        # Already black background
        black_background_sketch = gray

    # Resize to target size
    black_background_sketch = cv2.resize(black_background_sketch, target_size)

    # Apply threshold to ensure clear black/white distinction
    _, black_background_sketch = cv2.threshold(black_background_sketch, 50, 255, cv2.THRESH_BINARY)

    # Convert back to 3-channel RGB format
    black_background_sketch_rgb = np.stack([black_background_sketch] * 3, axis=-1)

    # Convert back to PIL Image
    return Image.fromarray(black_background_sketch_rgb)


def is_sketch_config(config_path):
    """
    Check if the config file contains 'sketch' in its name or path.

    Args:
        config_path (str): Path to the config file

    Returns:
        bool: True if it's a sketch configuration
    """
    config_name = Path(config_path).name.lower()
    return 'sketch' in config_name


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config = OmegaConf.load(args.config)
    samples = []

    # Check if this is a sketch configuration
    is_sketch = is_sketch_config(args.config)
    if is_sketch:
        print("🎨 Detected sketch configuration - will apply scribble preprocessing")

    # create validation pipeline
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet",
                                                       unet_additional_kwargs=OmegaConf.to_container(
                                                           inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""

            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get(
                "controlnet_additional_kwargs", {}))

            auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            if "controlnet" in controlnet_state_dict:
                controlnet_state_dict = controlnet_state_dict["controlnet"]
            elif "state_dict" in controlnet_state_dict:
                controlnet_state_dict = controlnet_state_dict["state_dict"]
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if
                                     "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            missing_keys, unexpected_keys = controlnet.load_state_dict(controlnet_state_dict, strict=False)
            print(f"ControlNet loading - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

            if missing_keys:
                print("Missing keys (first 10):")
                for key in missing_keys[:10]:
                    print(f"  - {key}")

            if unexpected_keys:
                print("Unexpected keys (first 10):")
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")

            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0),
                    ratio=(model_config.W / model_config.H, model_config.W / model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else:
                image_norm = lambda x: x

            processed_images = []
            for path in image_paths:
                if is_sketch:
                    print(f"🖌️  Processing sketch image: {path}")
                    # Apply scribble preprocessing (black bg, white lines)
                    processed_img = process_sketch_to_scribble(path, (model_config.W, model_config.H))
                    processed_images.append(image_norm(image_transforms(processed_img)))
                else:
                    # Use original processing
                    processed_images.append(image_norm(image_transforms(Image.open(path).convert("RGB"))))

            controlnet_images = processed_images

            # controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                if is_sketch:
                    # Save with special naming for sketch images
                    Image.fromarray((255. * (image.numpy().transpose(1, 2, 0))).astype(np.uint8)).save(
                        f"{savedir}/control_images/{i}_processed_sketch.png")
                else:
                    Image.fromarray((255. * (image.numpy().transpose(1, 2, 0))).astype(np.uint8)).save(
                        f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path=model_config.get("adapter_lora_path", ""),
            adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(
            model_config.n_prompt) == 1 else model_config.n_prompt

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

            # manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())

            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,

                controlnet_images=controlnet_images,
                controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            print(f"save to {savedir}/sample/{prompt}.gif")

            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="/workspace/AD-Evo-Tuner/models/StableDiffusion")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v5.yaml")
    parser.add_argument("--config", type=str, default="/workspace/AnimateDiff/PaintFlow/configs/prompts/3_sparsectrl/3_3_sparsectrl_sketch.yaml")

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)