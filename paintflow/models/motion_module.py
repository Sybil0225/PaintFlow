from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward

from einops import rearrange, repeat
import math



def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
        in_channels,
        motion_module_type: str,
        motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs, )
    elif motion_module_type == "PaintingEnhancedVanilla":
        return PaintingEnhancedVanillaMotionModule(in_channels=in_channels, **motion_module_kwargs)
    else:
        raise ValueError(f"Unknown motion module type: {motion_module_type}")


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,

        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,

                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,

                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout = 0.,
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 32,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0.,
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class PaintingEnhancedVanillaMotionModule(VanillaTemporalModule):

    def __init__(
            self,
            in_channels,
            num_attention_heads=8,
            num_transformer_block=2,
            attention_block_types=("Temporal_Self", "Temporal_Self"),
            cross_frame_attention_mode=None,
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=24,
            temporal_attention_dim_div=1,
            zero_initialize=True,
            enable_painting_enhancement=True,
            painting_stage_dim=64,
            progressive_strength=0.3,
            enable_progressive_attention=True,
            enforce_stage_progression=True,
            stage_transition_smoothness=0.8,
    ):
        super().__init__(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            num_transformer_block=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            temporal_attention_dim_div=temporal_attention_dim_div,
            zero_initialize=False,
        )

        self.enable_painting_enhancement = enable_painting_enhancement
        self.progressive_strength = progressive_strength
        self.enable_progressive_attention = enable_progressive_attention
        self.enforce_stage_progression = enforce_stage_progression
        self.stage_transition_smoothness = stage_transition_smoothness

        if enable_painting_enhancement:
            self.stage_embedding = nn.Embedding(4, painting_stage_dim)

            # Stage-specific feature transformers
            self.sketch_transformer = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
            )

            self.lineart_transformer = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
            )

            self.coloring_transformer = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
            )

            self.shading_transformer = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
            )

            # Progression controller for strict stage enforcement
            self.progression_controller = nn.Sequential(
                nn.Linear(painting_stage_dim + 1, painting_stage_dim // 2),
                nn.SiLU(),
                nn.Linear(painting_stage_dim // 2, 4),
                nn.Softmax(dim=-1)
            )

            self._enhance_temporal_transformer()

            if zero_initialize:
                self._zero_initialize_painting_components()

    def _enhance_temporal_transformer(self):
        """Add progress-aware attention control to temporal transformer"""
        for block in self.temporal_transformer.transformer_blocks:
            for attention in block.attention_blocks:
                if hasattr(attention, 'attention_mode') and attention.attention_mode == "Temporal":
                    query_dim = attention.to_q.in_features
                    attention.progress_adapter = nn.Sequential(
                        nn.Linear(query_dim, query_dim),
                        nn.LayerNorm(query_dim),
                        nn.SiLU(),
                        nn.Linear(query_dim, query_dim),
                    )

                    nn.init.zeros_(attention.progress_adapter[-1].weight)
                    nn.init.zeros_(attention.progress_adapter[-1].bias)

                    attention._original_forward = attention.forward
                    attention.forward = self._create_progress_aware_attention_forward(attention)

    def _create_progress_aware_attention_forward(self, attention):
        """Create progress-aware attention forward method"""

        def progress_aware_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
            batch_size, sequence_length, _ = hidden_states.shape

            if attention.attention_mode == "Temporal" and video_length > 1:
                d = hidden_states.shape[1]
                hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

                # Apply progressive feature transformation
                if hasattr(attention, 'progress_adapter'):
                    frame_weights = torch.linspace(0, 1, video_length, device=hidden_states.device)
                    frame_weights = frame_weights.view(1, video_length, 1).expand(hidden_states.shape[0], -1, -1)

                    progress_features = attention.progress_adapter(hidden_states)
                    hidden_states = hidden_states + progress_features * frame_weights * self.progressive_strength

                # Create strict causal mask for painting progression
                if attention_mask is None:
                    attention_mask = self._create_strict_painting_mask(video_length, hidden_states.device)

                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return attention._original_forward(hidden_states, encoder_hidden_states, attention_mask, video_length)

        return progress_aware_forward

    def _create_strict_painting_mask(self, video_length, device):
        """Create strict painting progression mask"""
        mask = torch.full((video_length, video_length), -1e9, device=device)

        for i in range(video_length):
            current_progress = i / max(video_length - 1, 1)

            for j in range(i + 1):
                past_progress = j / max(video_length - 1, 1)

                if j == i:
                    mask[i, j] = 0
                else:
                    stage_distance = self._calculate_stage_distance(past_progress, current_progress)
                    mask[i, j] = -0.1 * stage_distance

        return mask.unsqueeze(0)

    def _calculate_stage_distance(self, past_progress, current_progress):
        """Calculate distance between painting stages"""
        past_stage = int(past_progress * 4)
        current_stage = int(current_progress * 4)
        return abs(current_stage - past_stage)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        if not self.enable_painting_enhancement:
            return super().forward(input_tensor, temb, encoder_hidden_states, attention_mask, anchor_frame_idx)

        batch_size, channels, video_length, height, width = input_tensor.shape

        # Strict stage inference with monotonic progression
        stage_indices, stage_weights = self._infer_strict_painting_stages(video_length)
        stage_indices = stage_indices.to(input_tensor.device).long()
        stage_weights = stage_weights.to(input_tensor.device)

        # Apply stage-specific feature transformations
        transformed_input = self._apply_stage_specific_transforms(
            input_tensor, stage_indices, stage_weights
        )

        # Inject progress info to transformer
        self._inject_progress_info_to_transformer(stage_indices, stage_weights)

        # Process through enhanced transformer
        hidden_states = self.temporal_transformer(transformed_input, encoder_hidden_states, attention_mask)

        return hidden_states

    def _infer_strict_painting_stages(self, video_length):
        """Strict painting stage inference with monotonic progression"""
        stage_durations = torch.tensor([0.3, 0.25, 0.25, 0.2])  # sketch longer
        stage_boundaries = torch.cumsum(stage_durations, dim=0)

        time_progress = torch.linspace(0.01, 0.99, video_length)

        stage_indices = torch.zeros(video_length, dtype=torch.long)
        stage_weights = torch.zeros(video_length, 4)

        for i, progress in enumerate(time_progress):
            stage_idx = torch.searchsorted(stage_boundaries, progress, right=False)
            stage_idx = torch.clamp(stage_idx, 0, 3)

            stage_indices[i] = stage_idx

            # Smooth stage transitions while maintaining dominance
            if self.stage_transition_smoothness > 0 and i > 0:
                prev_stage = stage_indices[i - 1].item()
                curr_stage = stage_idx.item()

                if curr_stage > prev_stage:
                    transition_weight = self.stage_transition_smoothness
                    stage_weights[i, prev_stage] = transition_weight
                    stage_weights[i, curr_stage] = 1.0 - transition_weight
                else:
                    stage_weights[i, curr_stage] = 1.0
            else:
                stage_weights[i, stage_idx] = 1.0

        # Enforce monotonic progression - no stage regression
        for i in range(1, len(stage_indices)):
            if stage_indices[i] < stage_indices[i - 1]:
                stage_indices[i] = stage_indices[i - 1]
                stage_weights[i] = stage_weights[i - 1].clone()

        return stage_indices, stage_weights

    def _apply_stage_specific_transforms(self, input_tensor, stage_indices, stage_weights):
        """Apply stage-specific feature transformations"""
        batch_size, channels, video_length, height, width = input_tensor.shape

        # Reshape for processing
        input_flat = rearrange(input_tensor, "b c f h w -> (b f h w) c")
        transformed_flat = input_flat.clone()

        # Apply stage-specific transformations frame by frame
        for frame_idx in range(video_length):
            frame_start = frame_idx * batch_size * height * width
            frame_end = (frame_idx + 1) * batch_size * height * width
            frame_features = input_flat[frame_start:frame_end]

            weights = stage_weights[frame_idx]  # [4]

            transformed_frame = frame_features.clone()

            if weights[0] > 0:  # sketch stage
                sketch_features = self.sketch_transformer(frame_features)
                transformed_frame += (sketch_features - frame_features) * weights[0] * self.progressive_strength

            if weights[1] > 0:  # lineart stage
                lineart_features = self.lineart_transformer(frame_features)
                transformed_frame += (lineart_features - frame_features) * weights[1] * self.progressive_strength

            if weights[2] > 0:  # coloring stage
                coloring_features = self.coloring_transformer(frame_features)
                transformed_frame += (coloring_features - frame_features) * weights[2] * self.progressive_strength

            if weights[3] > 0:  # shading stage
                shading_features = self.shading_transformer(frame_features)
                transformed_frame += (shading_features - frame_features) * weights[3] * self.progressive_strength

            transformed_flat[frame_start:frame_end] = transformed_frame

        # Reshape back
        transformed_tensor = rearrange(transformed_flat, "(b f h w) c -> b c f h w",
                                       b=batch_size, f=video_length, h=height, w=width)

        return transformed_tensor

    def _inject_progress_info_to_transformer(self, stage_indices, stage_weights):
        """Inject progress info to transformer"""
        setattr(self.temporal_transformer, 'current_stage_indices', stage_indices)
        setattr(self.temporal_transformer, 'current_stage_weights', stage_weights)

    def _zero_initialize_painting_components(self):
        """Zero initialize painting enhancement components"""
        # Zero init stage transformers
        for transformer in [self.sketch_transformer, self.lineart_transformer,
                            self.coloring_transformer, self.shading_transformer]:
            for module in transformer:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Zero init progression controller
        for module in self.progression_controller:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero init temporal transformer proj_out
        self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def get_painting_progress_info(self, video_length):
        """Get painting progress info for debugging"""
        stage_indices, stage_weights = self._infer_strict_painting_stages(video_length)
        stage_names = ['sketch', 'lineart', 'coloring', 'shading']

        info = {
            'stage_indices': stage_indices.tolist(),
            'stage_names': [stage_names[i] for i in stage_indices],
            'stage_weights': stage_weights.tolist(),
            'progressive_strength': self.progressive_strength,
            'enforce_progression': self.enforce_stage_progression,
        }

        return info

    def load_vanilla_weights(self, checkpoint_path):
        """Load vanilla motion module weights"""
        print(f"Loading vanilla motion module weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        motion_state_dict = {}
        for key, value in state_dict.items():
            if 'motion_modules' in key:
                new_key = key.split('motion_modules.')[-1]
                if new_key.startswith('temporal_transformer'):
                    motion_state_dict[new_key] = value

        if not motion_state_dict:
            for key, value in state_dict.items():
                if key.startswith('temporal_transformer'):
                    motion_state_dict[key] = value

        if not motion_state_dict:
            print("Warning: No temporal_transformer weights found in checkpoint")
            return self

        missing_keys, unexpected_keys = self.load_state_dict(motion_state_dict, strict=False)

        print(f"Loaded vanilla motion module weights:")
        print(f"  Successfully loaded: {len(motion_state_dict)} parameters")
        print(f"  Missing keys: {len(missing_keys)} (expected for painting enhancement components)")
        print(f"  Unexpected keys: {len(unexpected_keys)}")

        return self


class PaintingEnhancedTemporalTransformer3DModel(TemporalTransformer3DModel):
    """增强版的TemporalTransformer3DModel，添加绘画过程感知"""

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # 获取当前的阶段信息（如果有的话）
        current_stage_emb = getattr(self, 'current_stage_emb', None)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                stage_embeddings=current_stage_emb
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class PaintingEnhancedTemporalTransformerBlock(TemporalTransformerBlock):
    """增强版的TemporalTransformerBlock"""

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None,
                stage_embeddings=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)

            # 如果attention支持绘画增强，传递阶段信息
            if hasattr(attention_block, 'painting_adapter'):
                hidden_states = attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                    stage_embeddings=stage_embeddings,
                ) + hidden_states
            else:
                hidden_states = attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                    video_length=video_length,
                ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        return hidden_states


# 为VersatileAttention添加绘画增强功能的扩展
def enhance_versatile_attention_for_painting(attention_module, progressive_strength=0.05):
    """为现有的VersatileAttention添加绘画感知能力"""

    original_forward = attention_module.forward

    def enhanced_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None,
                         stage_embeddings=None):
        # 如果有绘画适配器且是时序注意力
        if hasattr(attention_module, 'painting_adapter') and attention_module.attention_mode == "Temporal":
            batch_size, sequence_length, _ = hidden_states.shape
            d = hidden_states.shape[1]

            # 重排为时序格式
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

            # 应用绘画适配器
            if stage_embeddings is not None:
                # 使用阶段信息调制特征
                painting_features = attention_module.painting_adapter(hidden_states)

                # 计算阶段权重
                stage_weights = torch.softmax(stage_embeddings.mean(dim=-1), dim=-1)  # [B, F]
                stage_weights = stage_weights.unsqueeze(1).expand(-1, d, -1).reshape(-1, video_length, 1)

                # 应用阶段加权的绘画特征
                hidden_states = hidden_states + painting_features * stage_weights * progressive_strength

            # 创建渐进式注意力掩码
            if getattr(attention_module, 'enable_progressive_mask', False):
                if attention_mask is None:
                    attention_mask = create_progressive_painting_mask(video_length, hidden_states.device)

            # 重排回原格式
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        # 调用原始forward
        return original_forward(hidden_states, encoder_hidden_states, attention_mask, video_length)

    # 替换forward方法
    attention_module.forward = enhanced_forward
    return attention_module


def create_progressive_painting_mask(video_length, device):
    """创建适合绘画过程的渐进式注意力掩码"""
    mask = torch.zeros(video_length, video_length, device=device)

    for i in range(video_length):
        # 绘画进度：0-1
        progress = i / max(video_length - 1, 1)

        for j in range(i + 1):  # 只看过去和当前
            distance = i - j
            if distance == 0:
                mask[i, j] = 0  # 当前帧
            else:
                # 早期阶段更关注近邻，后期阶段能看到更远的历史
                locality_weight = -0.1 * distance * (1 - progress * 0.9)
                mask[i, j] = locality_weight

    # 强制因果性：未来帧不可见
    future_mask = torch.triu(torch.ones(video_length, video_length, device=device), diagonal=1) * -1e9
    mask = mask + future_mask

    return mask.unsqueeze(0)  # 添加batch维度


