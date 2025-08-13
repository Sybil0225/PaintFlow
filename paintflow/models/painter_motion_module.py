from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PainterMotionModule(nn.Module):
    """
    Enhanced motion module for painting process video generation.
    Key features:
    - Stage-aware temporal attention
    - Progressive painting process modeling
    - Fine-grained control over painting stages
    """

    def __init__(
            self,
            in_channels: int,
            num_attention_heads: int = 8,
            num_transformer_blocks: int = 2,
            attention_block_types: Tuple[str, ...] = ("Temporal_Self", "Temporal_Self"),
            cross_frame_attention_mode: Optional[str] = None,
            temporal_position_encoding: bool = True,
            temporal_position_encoding_max_len: int = 32,
            temporal_attention_dim_div: int = 1,
            num_painting_stages: int = 4,  # Typically: sketch, base, detail, final
            stage_embedding_dim: int = 128,
            zero_initialize: bool = True,
    ):
        super().__init__()

        self.num_stages = num_painting_stages
        self.in_channels = in_channels

        # Enhanced stage embedding with MLP projection
        self.stage_encoder = nn.Sequential(
            nn.Embedding(num_painting_stages, stage_embedding_dim),
            nn.SiLU(),
            nn.Linear(stage_embedding_dim, in_channels),
            nn.LayerNorm(in_channels)
        )

        # Stage-aware temporal transformer
        self.temporal_transformer = StageAwareTemporalTransformer(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            num_layers=num_transformer_blocks,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            temporal_attention_dim_div=temporal_attention_dim_div,
            num_stages=num_painting_stages
        )

        # Auxiliary stage predictor (optional, for training)
        self.stage_predictor = nn.Linear(in_channels, num_painting_stages) if zero_initialize else None

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(
            self,
            input_tensor: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stage_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = input_tensor.shape

        # Encode stage information
        if stage_idx is not None:
            # stage_idx shape: (batch_size,) containing values [0, num_stages-1]
            stage_emb = self.stage_encoder(stage_idx)  # (batch, in_channels)
            stage_emb = stage_emb.view(batch_size, -1, 1, 1, 1)
            input_tensor = input_tensor + stage_emb

        # Process through transformer
        hidden_states = self.temporal_transformer(
            input_tensor,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            stage_idx=stage_idx
        )

        return hidden_states


class StageAwareTemporalTransformer(nn.Module):
    """
    Temporal transformer with explicit painting stage conditioning
    """

    def __init__(
            self,
            in_channels: int,
            num_attention_heads: int,
            num_layers: int,
            attention_block_types: Tuple[str, ...],
            cross_frame_attention_mode: Optional[str],
            temporal_position_encoding: bool,
            temporal_position_encoding_max_len: int,
            temporal_attention_dim_div: int,
            num_stages: int,
            norm_num_groups: int = 32,
    ):
        super().__init__()

        inner_dim = num_attention_heads * (in_channels // num_attention_heads // temporal_attention_dim_div)

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            StageAwareTemporalBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=inner_dim // num_attention_heads,
                attention_block_types=attention_block_types,
                cross_frame_attention_mode=cross_frame_attention_mode,
                temporal_position_encoding=temporal_position_encoding,
                temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                num_stages=num_stages,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, stage_idx=None):
        # Rearrange for temporal processing
        batch, channel, num_frames, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = self.norm(hidden_states)

        # Project to inner dimension
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch * height * width, num_frames, channel)
        hidden_states = self.proj_in(hidden_states)

        # Process through stage-aware blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=num_frames,
                stage_idx=stage_idx
            )

        # Project back to original dimension
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, num_frames, channel).permute(0, 3, 4, 1, 2)
        hidden_states = hidden_states.contiguous()

        # Add residual and rearrange back
        output = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=num_frames) + residual
        return output


class StageAwareTemporalBlock(nn.Module):
    """
    Transformer block with painting stage conditioning at both attention and FFN
    """

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            attention_block_types: Tuple[str, ...],
            cross_frame_attention_mode: Optional[str],
            temporal_position_encoding: bool,
            temporal_position_encoding_max_len: int,
            num_stages: int,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.attention_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for block_type in attention_block_types:
            attention_block = StageAwareAttention(
                attention_mode=block_type.split("_")[0],
                cross_attention_dim=cross_attention_dim if "_Cross" in block_type else None,
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                cross_frame_attention_mode=cross_frame_attention_mode,
                temporal_position_encoding=temporal_position_encoding,
                temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                num_stages=num_stages,
            )
            self.attention_blocks.append(attention_block)
            self.norms.append(nn.LayerNorm(dim))

        # Stage-aware feed forward
        self.ff = StageAwareFeedForward(dim, num_stages, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, video_length=None, stage_idx=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                stage_idx=stage_idx
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states), stage_idx) + hidden_states
        return hidden_states


class StageAwareAttention(nn.Module):
    """
    Attention layer with painting stage conditioning
    """

    def __init__(
            self,
            attention_mode: str,
            cross_attention_dim: Optional[int],
            query_dim: int,
            heads: int,
            dim_head: int,
            dropout: float = 0.0,
            cross_frame_attention_mode: Optional[str] = None,
            temporal_position_encoding: bool = False,
            temporal_position_encoding_max_len: int = 32,
            num_stages: int = 4,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.attention_mode = attention_mode
        self.is_cross_attention = cross_attention_dim is not None

        # Stage projection layers
        self.stage_to_k = nn.Linear(num_stages, inner_dim)
        self.stage_to_v = nn.Linear(num_stages, inner_dim)

        # Standard attention components
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(cross_attention_dim if self.is_cross_attention else query_dim, inner_dim)
        self.to_v = nn.Linear(cross_attention_dim if self.is_cross_attention else query_dim, inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = None
        if temporal_position_encoding and attention_mode == "Temporal":
            self.pos_encoder = StageAwarePositionalEncoding(
                query_dim,
                num_stages=num_stages,
                dropout=dropout,
                max_len=temporal_position_encoding_max_len
            )

    def forward(self, hidden_states, encoder_hidden_states=None, video_length=None, stage_idx=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Prepare stage embedding
        stage_emb = None
        if stage_idx is not None:
            stage_onehot = F.one_hot(stage_idx, num_classes=self.num_stages).float()
            stage_emb_k = self.stage_to_k(stage_onehot)
            stage_emb_v = self.stage_to_v(stage_onehot)

        # Rearrange for temporal attention
        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states, stage_idx)

        # Project queries
        query = self.to_q(hidden_states)

        # Project keys and values
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Incorporate stage information
        if stage_emb is not None:
            key = key + stage_emb_k.unsqueeze(1)
            value = value + stage_emb_v.unsqueeze(1)

        # Attention computation
        attention_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )

        # Output projection
        output = self.to_out(attention_output)

        # Rearrange back if needed
        if self.attention_mode == "Temporal":
            output = rearrange(output, "(b d) f c -> (b f) d c", d=batch_size // video_length)

        return output


class StageAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that combines temporal and stage information
    """

    def __init__(self, d_model, num_stages, dropout=0.1, max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_stages = num_stages

        # Standard temporal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Stage encoding
        self.stage_embedding = nn.Embedding(num_stages, d_model)

    def forward(self, x, stage_idx=None):
        x = x + self.pe[:, :x.size(1)]
        if stage_idx is not None:
            stage_emb = self.stage_embedding(stage_idx).unsqueeze(1)
            x = x + stage_emb
        return self.dropout(x)


class StageAwareFeedForward(nn.Module):
    """
    Feed-forward network with stage conditioning
    """

    def __init__(self, dim, num_stages, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Stage conditioning
        self.stage_gate = nn.Sequential(
            nn.Embedding(num_stages, dim),
            nn.Sigmoid()
        )

    def forward(self, x, stage_idx=None):
        h = self.net(x)
        if stage_idx is not None:
            gate = self.stage_gate(stage_idx).unsqueeze(1)
            h = h * gate
        return h


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module