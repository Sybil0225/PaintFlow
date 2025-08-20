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
            painting_stage_dim=32,
            progressive_strength=0.05,
            enable_progressive_attention=True,
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

        if enable_painting_enhancement:
            self.stage_embedding = nn.Embedding(4, painting_stage_dim)

            self.stage_modulator = nn.Sequential(
                nn.Linear(painting_stage_dim, in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, in_channels),
            )

            self.progressive_weight_generator = nn.Sequential(
                nn.Linear(painting_stage_dim, in_channels // 4),
                nn.SiLU(),
                nn.Linear(in_channels // 4, 1),
                nn.Sigmoid()
            )

            self._enhance_temporal_transformer()

            if zero_initialize:
                self._zero_initialize_painting_components()

    def _enhance_temporal_transformer(self):
        for i, block in enumerate(self.temporal_transformer.transformer_blocks):
            for j, attention in enumerate(block.attention_blocks):
                if hasattr(attention, 'attention_mode') and attention.attention_mode == "Temporal":
                    if not hasattr(attention, 'painting_adapter'):
                        query_dim = attention.to_q.in_features
                        attention.painting_adapter = nn.Sequential(
                            nn.Linear(query_dim, query_dim // 4),
                            nn.SiLU(),
                            nn.Linear(query_dim // 4, query_dim),
                        )
                        nn.init.zeros_(attention.painting_adapter[-1].weight)
                        nn.init.zeros_(attention.painting_adapter[-1].bias)

                        attention.enable_progressive_mask = self.enable_progressive_attention
                        attention.progressive_strength = self.progressive_strength

    def _zero_initialize_painting_components(self):
        for module in self.stage_modulator:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        for module in self.progressive_weight_generator:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        if not self.enable_painting_enhancement:

            return super().forward(input_tensor, temb, encoder_hidden_states, attention_mask, anchor_frame_idx)

        batch_size, channels, video_length, height, width = input_tensor.shape


        stage_indices = self._infer_painting_stages(video_length)
        stage_indices = stage_indices.to(input_tensor.device).long()


        stage_emb = self.stage_embedding(stage_indices)  # [F, stage_dim]
        stage_emb = stage_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, F, stage_dim]


        stage_modulation = self.stage_modulator(stage_emb)  # [B, F, C]
        stage_modulation = stage_modulation.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, F, 1, 1]


        progressive_weights = self.progressive_weight_generator(stage_emb)  # [B, F, 1]
        progressive_weights = progressive_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, F, 1, 1]


        modulated_input = input_tensor + stage_modulation * self.progressive_strength


        self._inject_stage_info_to_transformer(stage_emb)


        hidden_states = modulated_input
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)


        output = hidden_states * (1.0 + progressive_weights * self.progressive_strength * 0.1)

        return output

    def _infer_painting_stages(self, video_length):
        """
        0-sketch, 1-line_art, 2-base_color, 3-shading
        """
        sketch_ratio = 0.2
        line_art_ratio = 0.3
        base_color_ratio = 0.30
        shading_ratio = 0.20

        sketch_frames = max(1, int(video_length * sketch_ratio))
        line_art_frames = max(1, int(video_length * line_art_ratio))
        base_color_frames = max(1, int(video_length * base_color_ratio))
        shading_frames = video_length - sketch_frames - line_art_frames - base_color_frames

        stages = []
        stages.extend([0] * sketch_frames)
        stages.extend([1] * line_art_frames)
        stages.extend([2] * base_color_frames)
        stages.extend([3] * shading_frames)

        return torch.tensor(stages, dtype=torch.long)


    def _inject_stage_info_to_transformer(self, stage_emb):

        if hasattr(self.temporal_transformer, 'current_stage_emb'):
            self.temporal_transformer.current_stage_emb = stage_emb
        else:
            setattr(self.temporal_transformer, 'current_stage_emb', stage_emb)

    def load_vanilla_weights(self, checkpoint_path):
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


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None,
                stage_embeddings=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)


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



def enhance_versatile_attention_for_painting(attention_module, progressive_strength=0.05):


    original_forward = attention_module.forward

    def enhanced_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None,
                         stage_embeddings=None):

        if hasattr(attention_module, 'painting_adapter') and attention_module.attention_mode == "Temporal":
            batch_size, sequence_length, _ = hidden_states.shape
            d = hidden_states.shape[1]


            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)


            if stage_embeddings is not None:

                painting_features = attention_module.painting_adapter(hidden_states)


                stage_weights = torch.softmax(stage_embeddings.mean(dim=-1), dim=-1)  # [B, F]
                stage_weights = stage_weights.unsqueeze(1).expand(-1, d, -1).reshape(-1, video_length, 1)


                hidden_states = hidden_states + painting_features * stage_weights * progressive_strength


            if getattr(attention_module, 'enable_progressive_mask', False):
                if attention_mask is None:
                    attention_mask = create_progressive_painting_mask(video_length, hidden_states.device)


            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)


        return original_forward(hidden_states, encoder_hidden_states, attention_mask, video_length)


    attention_module.forward = enhanced_forward
    return attention_module


def create_progressive_painting_mask(video_length, device):

    mask = torch.zeros(video_length, video_length, device=device)

    for i in range(video_length):
        # 0-1
        progress = i / max(video_length - 1, 1)

        for j in range(i + 1):
            distance = i - j
            if distance == 0:
                mask[i, j] = 0
            else:

                locality_weight = -0.1 * distance * (1 - progress * 0.9)
                mask[i, j] = locality_weight


    future_mask = torch.triu(torch.ones(video_length, video_length, device=device), diagonal=1) * -1e9
    mask = mask + future_mask

    return mask.unsqueeze(0)


