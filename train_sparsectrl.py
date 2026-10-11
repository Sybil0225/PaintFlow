import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as torch_F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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

from paintflow.data.dataset import  PaintCtrlDataset
from paintflow.models.unet import UNet3DConditionModel
from paintflow.pipelines.pipeline_animation import AnimationPipeline
from paintflow.utils.util import save_videos_grid, zero_rank_print
from paintflow.models.sparse_controlnet import SparseControlNetModel



def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

    return local_rank



def main(
    image_finetune: bool,

    name: str,
    use_wandb: bool,
    launcher: str,

    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,

    vae_checkpoint_path: str = "",
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    controlnet_checkpoint_path: str = "",
    motion_module_path: str = "",
    controlnet_additional_kwargs: Dict = {},
    sparse_ctrl_config: Dict = {},


    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,

    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 16,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # #Initialize single gpu training
    # local_rank = 0
    # global_rank = 0
    # num_processes = 1
    # is_main_process = True

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="paintflow", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))



    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    # vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # Load pretrained vae weights
    if vae_checkpoint_path != "":
        zero_rank_print(f"Loading custom VAE from: {vae_checkpoint_path}")

        try:
            if vae_checkpoint_path.endswith(".safetensors"):
                state_dict = load_file(vae_checkpoint_path, device="cpu")
                state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}

                vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
                missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

                zero_rank_print(f"VAE missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    zero_rank_print(f"Unexpected keys: {unexpected_keys[:5]}... (truncated)")

            elif os.path.isdir(vae_checkpoint_path):
                vae = AutoencoderKL.from_pretrained(vae_checkpoint_path)
            else:
                raise ValueError(f"Unsupported VAE format: {vae_checkpoint_path}")

            # 统一设备转移
            vae = vae.to(local_rank)

        except Exception as e:
            zero_rank_print(f"Failed to load custom VAE: {str(e)}")
            zero_rank_print("Falling back to default VAE")
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(local_rank)
    else:
        zero_rank_print("Using default VAE from pretrained model")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(local_rank)

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        if unet_checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(unet_checkpoint_path)
        else:
            state_dict = torch.load(unet_checkpoint_path, map_location="cpu")  # 继续用 torch.load() 处理 .pt/.pth

        if "global_step" in state_dict:
            zero_rank_print(f"global_step: {state_dict['global_step']}")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if not image_finetune:
        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None

        controlnet_config = OmegaConf.to_container(controlnet_additional_kwargs)

        controlnet = SparseControlNetModel.from_unet(
            unet,
            controlnet_additional_kwargs=controlnet_config
        )

        # === 步骤1: 加载预训练的 SparseCtrl 权重（仅 ControlNet 部分，不包含 motion module）===
        if controlnet_checkpoint_path != "":
            zero_rank_print(f"Loading SparseCtrl from checkpoint: {controlnet_checkpoint_path}")
            if controlnet_checkpoint_path.endswith(".safetensors"):
                state_dict = load_file(controlnet_checkpoint_path)
            else:
                state_dict = torch.load(controlnet_checkpoint_path, map_location="cpu")

            state_dict = state_dict["controlnet"] if "controlnet" in state_dict else state_dict
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            controlnet_only_state_dict = {}
            skipped_keys = []

            for name, param in state_dict.items():
                # 排除条件：motion_modules, pos_encoder.pe, animatediff_config
                if any(skip_pattern in name for skip_pattern in ["motion_modules", "pos_encoder.pe"]):
                    skipped_keys.append(name)
                    continue
                controlnet_only_state_dict[name] = param

            # 移除配置相关的键
            controlnet_only_state_dict.pop("animatediff_config", "")

            zero_rank_print(f"Skipped {len(skipped_keys)} motion module related keys")
            zero_rank_print(f"Loading {len(controlnet_only_state_dict)} ControlNet weights")

            m, u = controlnet.load_state_dict(controlnet_only_state_dict, strict=False)
            zero_rank_print(f"ControlNet base weights - Missing: {len(m)}, Unexpected: {len(u)}")

            # 打印部分跳过的键用于调试
            if skipped_keys:
                zero_rank_print(f"Sample skipped keys: {skipped_keys[:5]}...")

    # === 步骤2: 单独加载自定义 motion module 权重到 UNet 和 ControlNet ===
    if motion_module_path and os.path.exists(motion_module_path):
        zero_rank_print(f"Loading motion module from: {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")

        # 处理不同的 checkpoint 格式
        if "state_dict" in motion_module_state_dict:
            motion_module_state_dict = motion_module_state_dict["state_dict"]
        elif "model" in motion_module_state_dict:
            motion_module_state_dict = motion_module_state_dict["model"]

        # fliter and only load weights of motion module
        motion_module_keys = {k: v for k, v in motion_module_state_dict.items()
                              if "motion_module" in k}

        # 加载到 UNet
        if motion_module_keys:
            unet_missing, unet_unexpected = unet.load_state_dict(motion_module_keys, strict=False)
            print(f"UNet Motion Module - Missing: {len(unet_missing)}, Unexpected: {len(unet_unexpected)}")

        # 加载到 ControlNet
        if motion_module_keys:
            controlnet_missing, controlnet_unexpected = controlnet.load_state_dict(motion_module_keys, strict=False)
            print(
                f"ControlNet Motion Module - Missing: {len(controlnet_missing)}, Unexpected: {len(controlnet_unexpected)}")

    else:
        print("No motion module path specified or file not found, using initialized weights")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)
    trainable_params = list(controlnet.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = PaintCtrlDataset(
    video_dir=train_data.get("video_folder", "/workspace/video_ctrl/videos"),
    annotations_path=train_data.get("csv_path", "/workspace/final_frame/annotation_ctrl.csv"),
    sketch2video_path=train_data.get("video_json_path", "/workspace/video_ctrl/sketch2video_dataset.jsonl"),
    condition_type="sketch",
    resolution=(train_data.get("sample_size", 256), train_data.get("sample_size", 256)),
        split_file=train_data.get("split_file", None),

    )

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    controlnet.to(local_rank)
    controlnet = DDP(controlnet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    # 获取 SparseCtrl 配置
    enable_random_masking = sparse_ctrl_config.get("enable_random_masking", True)
    min_condition_frames = sparse_ctrl_config.get("min_condition_frames", 1)
    max_condition_frames = sparse_ctrl_config.get("max_condition_frames", -1)
    mask_probability = sparse_ctrl_config.get("mask_probability", 1.0)

    for epoch in range(first_epoch, num_train_epochs):
        print(f"Starting epoch {epoch}")
        train_dataloader.sampler.set_epoch(epoch)
        controlnet.train()
        unet.eval()

        for step, batch in enumerate(train_dataloader):
            print(f"Processing batch {step}...")

            # get control image and video
            control_image = batch["sketch"].to(local_rank)  # [B, C, H, W]
            pixel_values = batch["video"].to(local_rank)  # [B, F, C, H, W]

            # generate corresponding prompt
            batch_text = ["painting process, sketch to final artwork"] * control_image.shape[0]

            if cfg_random_null_text:
                batch_text = [text if random.random() > cfg_random_null_text_ratio else "" for text in batch_text]


            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                # save control image
                for idx, sketch in enumerate(control_image.cpu()):
                    sketch_normalized = sketch / 2. + 0.5
                    torchvision.utils.save_image(sketch_normalized,
                                                 f"{output_dir}/sanity_check/sketch-{global_rank}-{idx}.png")
                # save video frame
                for idx, video in enumerate(pixel_values.cpu()):
                    video_normalized = video / 2. + 0.5
                    for frame_idx, frame in enumerate(video):
                        torchvision.utils.save_image(frame,
                                                     f"{output_dir}/sanity_check/video-{global_rank}-{idx}-frame{frame_idx}.png")

            ### >>>> Training >>>> ###

            # Convert videos to latent space
            video_length = pixel_values.shape[1]
            # control_image = control_image.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # [B, F, C, H, W]
            if len(control_image.shape) == 4:  # [B, C, H, W]
                # 在第2个维度（C之后）插入帧维度，然后重复
                control_image = control_image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)  # [B, C, F, H, W]


            with torch.no_grad():
                # process video frame
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # process control image
                # control_image = control_image.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # [B, F, C, H, W]
                # control_image = rearrange(control_image, "b f c h w -> b c f h w")

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch_text, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"

                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            target = noise # epsilon prediction

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # 实现 SparseCtrl 的随机掩码策略
                B, C, F, H, W = control_image.shape

                with torch.no_grad():
                    # 将 control_image 重新整形为 [B*F, C, H, W] 进行 VAE 编码
                    control_image_reshaped = rearrange(control_image, "b c f h w -> (b f) c h w")
                    # 编码到 latent space
                    control_latents = vae.encode(control_image_reshaped).latent_dist.sample()
                    # 重新整形回 [B, C_latent, F, H_latent, W_latent]
                    control_latents = rearrange(control_latents, "(b f) c h w -> b c f h w", f=F)
                    control_latents = control_latents * 0.18215  # 同样的缩放因子

                B_latent, C_latent, F_latent, H_latent, W_latent = control_latents.shape

                if enable_random_masking and random.random() < mask_probability:
                        # 为每个batch样本创建随机掩码
                        conditioning_mask = torch.zeros(B_latent, 1, F_latent, H_latent, W_latent,
                                                        device=control_latents.device, dtype=control_latents.dtype)

                        for b in range(B_latent):
                            # 确定最大条件帧数
                            actual_max_frames = F_latent if max_condition_frames == -1 else min(max_condition_frames, F_latent)
                            # 随机选择 Nc 个帧接受条件信号 (min_condition_frames <= Nc <= actual_max_frames)
                            Nc = random.randint(min_condition_frames, actual_max_frames)
                            # 随机选择 Nc 个帧索引（不重复）
                            selected_frames = random.sample(range(F), Nc)
                            # 设置选中帧的mask为1
                            conditioning_mask[b, 0, selected_frames, :, :] = 1.0

                        # 应用掩码到控制信号
                        masked_control_latents = control_latents * conditioning_mask
                else:
                    # 不使用掩码，所有帧都有条件
                    conditioning_mask = torch.ones(B_latent, 1, F_latent, H_latent, W_latent, device=control_latents.device,
                                                   dtype=control_image.dtype)
                    masked_control_latents = control_latents

                # 通过 ControlNet 获取条件特征
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=masked_control_latents,
                    conditioning_mask=conditioning_mask,
                    return_dict=False,
                )

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,).sample
                loss = torch_F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                controlnet_dict = {k: v for k, v in controlnet.module.state_dict().items()}
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": controlnet_dict,
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"sparsectrl-sketch-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"sketchctrl.ckpt"))
                logging.info(f"Saved SparseCtrl state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                # SparseCtrl 的验证需要控制信号，这里先跳过或者简化
                # 可以在后续添加带控制信号的验证逻辑
                logging.info(f"Validation at step {global_step} - SparseCtrl training in progress")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="/workspace/AnimateDiff/PaintFlow/configs/training/v1/train_sketch_ctrl.yaml")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)