"""
v7.0: ConvTransformer World Model

Hybrid architecture combining:
- Efficient attention (windowed + axial) for global reasoning
- Depth-wise convolutions for local features
- AdaLN-Zero action conditioning with separate camera/movement pathways
- Spatial cross-attention for temporal context (preserves spatial structure)

Design targets:
- ~11M parameters
- ~16 FPS on M3 MacBook Air (MPS backend)
- Addresses ConvGRU failures: action dimensionality bottleneck, temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint
from typing import Optional, List, Tuple


# ----------------------------
# Utility Modules
# ----------------------------

class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA, PaLM). More expressive than ReLU/SiLU."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class DepthwiseSeparableConv(nn.Module):
    """
    Efficient convolution: depth-wise (spatial) + point-wise (channel mixing).
    ANE-accelerated on Apple Silicon.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw_conv(x)
        return x + residual


# ----------------------------
# Efficient Attention
# ----------------------------

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Partition spatial tensor into non-overlapping windows.

    x: (B, H, W, C)
    Returns: (num_windows * B, window_size * window_size, C), H, W
    """
    B, H, W, C = x.shape
    # Pad if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, Hp, Wp


def window_reverse(windows: torch.Tensor, window_size: int, Hp: int, Wp: int, H: int, W: int, B: int) -> torch.Tensor:
    """
    Reverse window partition.

    windows: (num_windows * B, window_size * window_size, C)
    Returns: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
    # Remove padding
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class EfficientSpatialAttention(nn.Module):
    """
    Hybrid attention: windowed (local) + axial (global).

    For 18x32 latent grid:
    - Window size 4x4: 16 tokens per window, captures local block patterns
    - Axial attention: attend along rows (32) and columns (18) for global connectivity

    Complexity: O(n * (w^2 + h + w)) vs O(n^2) for full attention
    ~39K attention entries vs 331K (8.5x reduction)
    """
    def __init__(self, dim: int = 384, num_heads: int = 6, window_size: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Windowed attention (local)
        self.window_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.window_proj = nn.Linear(dim, dim)

        # Relative position bias for windowed attention
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Axial attention (global, linear complexity)
        self.row_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.row_proj = nn.Linear(dim, dim)

        self.col_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.col_proj = nn.Linear(dim, dim)

        # Layer norms
        self.norm_window = nn.LayerNorm(dim)
        self.norm_row = nn.LayerNorm(dim)
        self.norm_col = nn.LayerNorm(dim)

    def _windowed_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply windowed self-attention."""
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        # Partition into windows
        x_windows, Hp, Wp = window_partition(x, self.window_size)  # (num_win*B, win_size^2, C)

        # QKV
        qkv = self.window_qkv(x_windows)
        qkv = qkv.reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, num_win*B, heads, win_size^2, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with relative position bias
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out = self.window_proj(out)

        # Reverse windows
        out = window_reverse(out, self.window_size, Hp, Wp, H, W, B)
        return out.view(B, N, C)

    def _row_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply row-wise attention (horizontal global connectivity)."""
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        # Reshape for row attention: (B*H, W, C)
        x_rows = x.view(B * H, W, C)

        # QKV
        qkv = self.row_qkv(x_rows)
        qkv = qkv.reshape(B * H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H, heads, W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * H, W, C)
        out = self.row_proj(out)

        return out.view(B, H, W, C).view(B, N, C)

    def _col_attention(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply column-wise attention (vertical global connectivity)."""
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        # Reshape for col attention: (B*W, H, C)
        x_cols = x.permute(0, 2, 1, 3).reshape(B * W, H, C)

        # QKV
        qkv = self.col_qkv(x_cols)
        qkv = qkv.reshape(B * W, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*W, heads, H, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * W, H, C)
        out = self.col_proj(out)

        return out.view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, N, C)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) where N = H * W
        Returns: (B, N, C)
        """
        # Windowed attention (local patterns)
        x = x + self._windowed_attention(self.norm_window(x), H, W)

        # Axial attention (global connectivity)
        x = x + self._row_attention(self.norm_row(x), H, W)
        x = x + self._col_attention(self.norm_col(x), H, W)

        return x


# ----------------------------
# Action Conditioning
# ----------------------------

class ActionAdaLN(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero).

    Inspired by DiT (Peebles & Xie, 2023).

    Key improvements over FiLM:
    - Separate camera/movement pathways (orthogonal gradients)
    - Gating mechanism (learned action routing)
    - Zero-init for stable training start (no early "action flash")
    """
    def __init__(self, hidden_dim: int, action_dim: int = 15):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Separate pathways (like SeparateActionFiLM from v6.1)
        # Camera: yaw(5) + pitch(3) = 8 dims
        self.camera_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 3 * hidden_dim)  # scale, shift, gate
        )

        # Movement: WASD(4) + jump(1) + sprint(1) + sneak(1) = 7 dims
        self.movement_mlp = nn.Sequential(
            nn.Linear(7, 128),
            nn.SiLU(),
            nn.Linear(128, 3 * hidden_dim)
        )

        # Zero-initialize final projections for stable training
        nn.init.zeros_(self.camera_mlp[-1].weight)
        nn.init.zeros_(self.camera_mlp[-1].bias)
        nn.init.zeros_(self.movement_mlp[-1].weight)
        nn.init.zeros_(self.movement_mlp[-1].bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C) or (B, C, H, W)
        action: (B, 15)

        Returns: (normalized_x, gate)
        """
        camera = action[..., :8]
        movement = action[..., 8:]

        # Combine orthogonal contributions
        cam_params = self.camera_mlp(camera)
        mov_params = self.movement_mlp(movement)
        params = cam_params + mov_params  # (B, 3*C)

        scale, shift, gate = params.chunk(3, dim=-1)

        # Reshape for broadcast
        if x.dim() == 3:
            # (B, N, C)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            gate = torch.sigmoid(gate).unsqueeze(1)
        else:
            # (B, C, H, W)
            scale = scale.view(-1, self.hidden_dim, 1, 1)
            shift = shift.view(-1, self.hidden_dim, 1, 1)
            gate = torch.sigmoid(gate).view(-1, self.hidden_dim, 1, 1)

        # AdaLN: scale and shift the input
        x_normalized = x * (1 + scale) + shift

        return x_normalized, gate

    @torch.no_grad()
    def action_to_params(self, action: torch.Tensor, *, split_paths: bool = False) -> dict:
        """
        Diagnostic helper: compute AdaLN params induced by `action` without needing `x`.

        Returns dict with keys:
          - combined: scale/shift/gate_raw/gate (B, C)
          - (optional) camera/movement: same keys per path (pre-sigmoid gate_raw)
        """
        camera = action[..., :8]
        movement = action[..., 8:]

        cam_params = self.camera_mlp(camera)
        mov_params = self.movement_mlp(movement)
        params = cam_params + mov_params

        scale, shift, gate_raw = params.chunk(3, dim=-1)
        out = {
            "combined": {
                "scale": scale,
                "shift": shift,
                "gate_raw": gate_raw,
                "gate": torch.sigmoid(gate_raw),
            }
        }

        if split_paths:
            cam_scale, cam_shift, cam_gate_raw = cam_params.chunk(3, dim=-1)
            mov_scale, mov_shift, mov_gate_raw = mov_params.chunk(3, dim=-1)
            out["camera"] = {
                "scale": cam_scale,
                "shift": cam_shift,
                "gate_raw": cam_gate_raw,
                "gate": torch.sigmoid(cam_gate_raw),
            }
            out["movement"] = {
                "scale": mov_scale,
                "shift": mov_shift,
                "gate_raw": mov_gate_raw,
                "gate": torch.sigmoid(mov_gate_raw),
            }

        return out


# ----------------------------
# Temporal Context
# ----------------------------

class TemporalCrossAttention(nn.Module):
    """
    Spatial cross-attention to past frames.

    Key improvement over v6.1's LightweightTemporalAttention:
    - Preserves spatial structure (48 tokens per past frame vs 1 pooled vector)
    - Enables direct spatial comparison for movement detection
    - 1-hop gradient path instead of recurrent bottleneck

    Memory: 4 frames x 48 tokens x 192D = ~37K floats = 148KB
    """
    def __init__(
        self,
        hidden_dim: int = 384,
        context_len: int = 4,
        downsample_factor: int = 3,
        num_heads: int = 4,
        recency_decay: float = 1.0  # NEW: 1.0=no decay, 0.9=prefer recent frames
    ):
        super().__init__()
        self.context_len = context_len
        self.downsample_factor = downsample_factor
        self.compressed_dim = hidden_dim // 2
        self.recency_decay = recency_decay  # NEW: exponential decay for old frames

        # Spatial downsampling: 18x32 -> 6x11 (with stride 3)
        # Actually computes to: floor(18/3) x floor(32/3) = 6 x 10 = 60 tokens
        self.compress = nn.Sequential(
            nn.Conv2d(hidden_dim, self.compressed_dim, kernel_size=3, stride=downsample_factor, padding=1),
            nn.SiLU(),
        )

        # Expand compressed context back to hidden_dim for attention
        self.context_proj = nn.Linear(self.compressed_dim, hidden_dim)

        # Cross-attention: current frame queries past frames
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def compress_frame(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compress spatial hidden state for temporal buffer.

        h: (B, C, H, W) -> (B, num_tokens, compressed_dim)
        """
        h_down = self.compress(h)  # (B, compressed_dim, H', W')
        return h_down.flatten(2).transpose(1, 2)  # (B, H'*W', compressed_dim)

    def forward(
        self,
        x: torch.Tensor,
        temporal_buffer: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        x: (B, N, C) current frame features
        temporal_buffer: list of (B, num_tokens, compressed_dim) compressed past frames

        Returns: (B, N, C)
        """
        if temporal_buffer is None or len(temporal_buffer) == 0:
            return x

        # Stack and project past frames
        recent = temporal_buffer[-self.context_len:] if self.context_len > 0 else []
        if len(recent) == 0:
            return x

        context = torch.cat(recent, dim=1)  # (B, total_past_tokens, compressed_dim)
        context = self.context_proj(context)  # (B, total_past_tokens, hidden_dim)

        # NEW: Apply exponential decay mask based on frame age
        if self.recency_decay < 1.0:
            frames_in_buffer = len(recent)
            tokens_per_frame = recent[0].shape[1]  # Each compressed frame has same number of tokens

            # Create decay weights: newest frame (index -1) gets weight 1.0,
            # older frames get exponentially decayed weights
            decay_weights = []
            for i in range(frames_in_buffer):
                age = frames_in_buffer - i - 1  # 0=newest, 7=oldest
                weight = self.recency_decay ** age
                decay_weights.extend([weight] * tokens_per_frame)

            decay_mask = torch.tensor(decay_weights, device=context.device, dtype=context.dtype)
            decay_mask = decay_mask.unsqueeze(0).unsqueeze(2)  # (1, total_past_tokens, 1)

            # Scale context by decay mask
            context = context * decay_mask  # (B, total_past_tokens, hidden_dim)

        # Cross-attend: current frame queries past frames
        x_norm = self.norm(x)
        out, _ = self.cross_attn(x_norm, context, context)

        return x + out  # Residual

    @torch.no_grad()
    def attention_stats(
        self,
        x: torch.Tensor,
        temporal_buffer: Optional[List[torch.Tensor]],
        *,
        num_query_tokens: int = 64,
    ) -> Optional[dict]:
        """
        Diagnostic helper: sample a subset of query tokens and compute attention weight stats.

        Returns:
          dict with:
            - attn_mean_per_src (src_len,)
            - attn_entropy (scalar)
            - src_len (int)
            - num_queries (int)
        """
        if temporal_buffer is None or len(temporal_buffer) == 0 or self.context_len <= 0:
            return None

        recent = temporal_buffer[-self.context_len:]
        if len(recent) == 0:
            return None

        context = torch.cat(recent, dim=1)  # (B, src_len, compressed_dim)
        context = self.context_proj(context)  # (B, src_len, hidden_dim)

        B, tgt_len, _ = x.shape
        src_len = context.shape[1]
        frames_in_buffer = len(recent)
        tokens_per_frame = int(recent[0].shape[1]) if frames_in_buffer > 0 else 0

        if tgt_len <= 0 or src_len <= 0:
            return None

        num_q = min(int(num_query_tokens), int(tgt_len))
        # Use same query indices for the whole batch (good enough for diagnostics).
        q_idx = torch.randperm(tgt_len, device=x.device)[:num_q]

        x_q = self.norm(x[:, q_idx, :])
        _, attn_w = self.cross_attn(x_q, context, context, need_weights=True, average_attn_weights=True)
        # attn_w: (B, num_q, src_len)
        attn_mean_per_src = attn_w.mean(dim=(0, 1))  # (src_len,)
        attn_mean_per_src = attn_mean_per_src / (attn_mean_per_src.sum() + 1e-9)

        # Entropy over src tokens for the averaged distribution.
        ent = -(attn_mean_per_src * torch.log(attn_mean_per_src + 1e-9)).sum()

        attn_mean_per_frame = None
        if frames_in_buffer > 0 and tokens_per_frame > 0 and frames_in_buffer * tokens_per_frame == src_len:
            attn_mean_per_frame = attn_mean_per_src.view(frames_in_buffer, tokens_per_frame).sum(dim=1)

        return {
            "attn_mean_per_src": attn_mean_per_src.detach(),
            "attn_mean_per_frame": attn_mean_per_frame.detach() if attn_mean_per_frame is not None else None,
            "attn_entropy": ent.detach(),
            "src_len": int(src_len),
            "num_queries": int(num_q),
            "frames_in_buffer": int(frames_in_buffer),
            "tokens_per_frame": int(tokens_per_frame),
        }


# ----------------------------
# Transformer Block
# ----------------------------

class ConvTransformerBlock(nn.Module):
    """
    Single transformer block with:
    - Depth-wise conv (local features, ANE accelerated)
    - Efficient spatial attention (windowed + axial)
    - AdaLN-Zero action conditioning
    - SwiGLU FFN
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        action_dim: int = 15,
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        self.dim = dim

        # Local conv processing
        self.conv = DepthwiseSeparableConv(dim)

        # Efficient spatial attention
        self.attn = EfficientSpatialAttention(dim, num_heads, window_size)

        # Action conditioning (AdaLN-Zero)
        self.adaln_attn = ActionAdaLN(dim, action_dim)
        self.adaln_ffn = ActionAdaLN(dim, action_dim)

        # FFN with SwiGLU (3x expansion, then halved by SwiGLU)
        mlp_hidden = int(dim * mlp_ratio * 2)  # 2x for SwiGLU split
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            SwiGLU(),
            nn.Linear(mlp_hidden // 2, dim),
        )

        # Zero-init FFN output for residual stability
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, action: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C) where N = H * W
        action: (B, 15)

        Returns: (B, N, C)
        """
        B, N, C = x.shape

        # Conv path (reshape to spatial)
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        x_spatial = self.conv(x_spatial)
        x = x_spatial.flatten(2).transpose(1, 2)  # Back to (B, N, C)

        # Attention with action conditioning
        x_norm, gate = self.adaln_attn(self.norm1(x), action)
        x = x + gate * self.attn(x_norm, H, W)

        # FFN with action conditioning
        x_norm, gate = self.adaln_ffn(self.norm2(x), action)
        x = x + gate * self.ffn(x_norm)

        return x


# ----------------------------
# Inverse Dynamics Module
# ----------------------------

class InverseDynamicsModule(nn.Module):
    """
    IDM adapted for transformer architecture.
    Predicts cumulative action from feature transitions.
    """
    def __init__(self, hidden_dim: int = 384, action_dim: int = 15, max_span: int = 5):
        super().__init__()

        # Spatial compression: use same as temporal compression
        self.compress = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  # -> 9x16
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # -> 5x8
            nn.SiLU(),
            nn.Flatten()
        )

        flattened_dim = 64 * 5 * 8  # 2560
        self.dt_embed = nn.Embedding(max_span + 1, 64)

        input_dim = (flattened_dim * 2) + 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim)
        )

    def forward(
        self,
        h_start: torch.Tensor,
        h_end: torch.Tensor,
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        h_start, h_end: (B, C, H, W) hidden states
        dt: (B,) time delta

        Returns: (B, action_dim) predicted action
        """
        z_start = self.compress(h_start)
        z_end = self.compress(h_end)
        t_emb = self.dt_embed(dt)

        combined = torch.cat([z_start, z_end, t_emb], dim=1)
        return self.mlp(combined)


# ----------------------------
# Full Model
# ----------------------------

class MinecraftConvTransformer(nn.Module):
    """
    v7.0: Hybrid ConvTransformer World Model.

    Architecture:
    - Token embedding + Conv stem
    - 4 ConvTransformer blocks with efficient attention
    - Temporal cross-attention to past frames
    - AdaLN-Zero action conditioning throughout

    Target specs:
    - ~11M parameters
    - ~16 FPS on M3 MacBook Air (MPS)
    - Addresses ConvGRU failures in temporal modeling and action conditioning
    """
    def __init__(
        self,
        codebook_size: int = 2048,
        embed_dim: int = 256,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        H: int = 18,
        W: int = 32,
        action_dim: int = 15,
        temporal_context_len: int = 4,
        window_size: int = 4,
        idm_max_span: int = 5,
        mlp_ratio: float = 2.0,
        use_checkpointing: bool = False,
        recency_decay: float = 1.0,  # NEW: Exponential decay for temporal attention (1.0=no decay)
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.H, self.W = H, W
        self.hidden_dim = hidden_dim
        self.temporal_context_len = temporal_context_len
        self.use_checkpointing = use_checkpointing

        # Token embedding
        self.embed = nn.Embedding(codebook_size, embed_dim)

        # Conv stem: project to hidden dim + initial spatial processing
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            DepthwiseSeparableConv(hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ConvTransformerBlock(
                hidden_dim, num_heads, window_size, action_dim, mlp_ratio
            ) for _ in range(num_layers)
        ])

        # Temporal cross-attention
        self.temporal_attn = TemporalCrossAttention(
            hidden_dim, temporal_context_len, downsample_factor=3, num_heads=4,
            recency_decay=recency_decay  # NEW: Pass decay parameter
        )

        # IDM for auxiliary supervision
        self.idm = InverseDynamicsModule(hidden_dim, action_dim, idm_max_span)

        # Output head
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, codebook_size)

        # Zero-init output for stable training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def get_features_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial features for IDM.

        x: (B, N, C) transformer features
        Returns: (B, C, H, W) spatial features
        """
        B = x.shape[0]
        return x.transpose(1, 2).reshape(B, self.hidden_dim, self.H, self.W)

    def step(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        temporal_buffer: Optional[List[torch.Tensor]] = None,
        *,
        return_spatial_features: bool = False,
    ):
        """
        Single-step inference.

        Args:
            z_t: (B, H, W) token indices
            action: (B, 15) action vector
            temporal_buffer: list of compressed past frames

        Returns:
            logits: (B, codebook_size, H, W)
            new_temporal_state: (B, num_tokens, compressed_dim) for buffer
            x_spatial: (B, C, H, W) post-temporal features (optional)
        """
        B = z_t.shape[0]

        # Embed tokens
        x = self.embed(z_t)  # (B, H, W, E)
        x = x.permute(0, 3, 1, 2)  # (B, E, H, W)

        # Conv stem
        x = self.stem(x)  # (B, C, H, W)

        # Flatten for transformer blocks
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Apply transformer blocks
        for block in self.blocks:
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, action, self.H, self.W, use_reentrant=False)
            else:
                x = block(x, action, self.H, self.W)

        # Temporal cross-attention
        x = self.temporal_attn(x, temporal_buffer)

        # Output projection
        x_out = self.out_norm(x)
        logits = self.out_proj(x_out)  # (B, N, codebook_size)

        # Reshape to spatial
        logits = logits.transpose(1, 2).reshape(B, self.codebook_size, self.H, self.W)

        # Compress current frame for temporal buffer
        x_spatial = x.transpose(1, 2).reshape(B, self.hidden_dim, self.H, self.W)
        new_temporal_state = self.temporal_attn.compress_frame(x_spatial)

        if return_spatial_features:
            return logits, new_temporal_state, x_spatial
        return logits, new_temporal_state

    def forward(
        self,
        Z_seq: torch.Tensor,
        A_seq: torch.Tensor,
        return_all: bool = False,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Unroll over sequence.

        Args:
            Z_seq: (B, K, H, W) token sequences
            A_seq: (B, K, 15) action sequences
            return_all: return all timestep logits
            return_features: return features for IDM

        Returns:
            logits: (B, codebook_size, H, W) or list if return_all
        """
        B, K, H, W = Z_seq.shape

        logits_list = []
        features_list = [] if return_features else None
        temporal_buffer = []

        for k in range(K):
            if return_features:
                logits, new_state, x_spatial = self.step(
                    Z_seq[:, k],
                    A_seq[:, k],
                    temporal_buffer,
                    return_spatial_features=True,
                )
                features_list.append(x_spatial)
            else:
                logits, new_state = self.step(Z_seq[:, k], A_seq[:, k], temporal_buffer)
            logits_list.append(logits)

            # Update temporal buffer
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > self.temporal_context_len:
                temporal_buffer.pop(0)

        if return_features:
            return logits_list if return_all else logits_list[-1], features_list

        if return_all:
            return logits_list
        return logits_list[-1]

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts['embed'] = sum(p.numel() for p in self.embed.parameters())
        counts['stem'] = sum(p.numel() for p in self.stem.parameters())
        counts['blocks'] = sum(p.numel() for p in self.blocks.parameters())
        counts['temporal_attn'] = sum(p.numel() for p in self.temporal_attn.parameters())
        counts['idm'] = sum(p.numel() for p in self.idm.parameters())
        counts['output'] = sum(p.numel() for p in self.out_norm.parameters()) + \
                          sum(p.numel() for p in self.out_proj.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
