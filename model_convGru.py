import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Core building blocks
# ----------------------------

class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.zr_gate = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=p)
        self.h_tilde_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=p)

    def forward(self, x, h):
        combined_zr = torch.cat([x, h], dim=1)
        zr = self.zr_gate(combined_zr)
        z, r = zr.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_h = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.h_tilde_gate(combined_h))

        h_new = (1 - z) * h + z * h_tilde
        return h_new


class SeparateActionFiLM(nn.Module):
    """
    v6.1: Separate FiLM pathways for camera (dims 0-7) and movement (dims 8-14).
    Prevents gradient competition between camera and movement actions.

    Oasis analysis: When camera and movement share weights,
    camera gradients (stronger LPIPS signal) dominate and suppress movement learning.
    """
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

        # Camera pathway: yaw(5) + pitch(3) = 8 dims
        self.camera_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8, 128),
                nn.SiLU(),
                nn.Linear(128, 2 * hidden_dim)
            ) for _ in range(n_layers)
        ])

        # Movement pathway: WASD(4) + jump(1) + sprint(1) + sneak(1) = 7 dims
        self.movement_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(7, 128),
                nn.SiLU(),
                nn.Linear(128, 2 * hidden_dim)
            ) for _ in range(n_layers)
        ])

    def forward(self, a):
        """
        a: (B, 15) or (B, T, 15)
        Returns: (gammas, betas) as stacked tensors (L, B, [T], C, 1, 1)
        """
        camera = a[..., :8]    # yaw + pitch bins
        movement = a[..., 8:]  # WASD + jump + sprint + sneak

        gammas = []
        betas = []

        for cam_mlp, mov_mlp in zip(self.camera_mlps, self.movement_mlps):
            cam_gb = cam_mlp(camera)
            mov_gb = mov_mlp(movement)

            # Combine additively (orthogonal contributions)
            gb = cam_gb + mov_gb
            g, b = gb.chunk(2, dim=-1)

            # Reshape for broadcast: (B, [T], C) -> (B, [T], C, 1, 1)
            g = g.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)

            gammas.append(g)
            betas.append(b)

        gammas = torch.stack(gammas)
        betas = torch.stack(betas)

        # Clamp to prevent "Action Flash" artifacts
        gammas = torch.clamp(gammas, -3.0, 3.0)
        betas = torch.clamp(betas, -3.0, 3.0)

        return gammas, betas


class LightweightTemporalAttention(nn.Module):
    """
    v6.1: Causal temporal attention over recent frames.
    Inspired by Oasis but optimized for inference speed.

    Key insight: Movement detection requires comparing frame t-k to frame t directly.
    RNN bottleneck compresses this through hidden state, losing the signal.
    This module provides direct temporal comparison with minimal overhead.

    Design choices for speed:
    - Spatial pooling before attention (18x32 -> 1x1)
    - Limited context length (default 8 frames)
    - Single attention layer with residual connection
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, context_len: int = 8):
        super().__init__()
        self.context_len = context_len
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        # Spatial pooling first (18x32 -> 1x1) for efficiency
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # QKV projection
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # Relative position bias (per head) applied to attention logits.
        # This avoids injecting position information into V directly (which can act like a shortcut).
        self.rel_pos_bias = nn.Embedding(2 * context_len + 1, num_heads)

    def pool_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        Pool a spatial hidden state to a vector for cheap temporal context storage.
        Accepts either (B, C, H, W) or already-pooled (B, C).
        """
        if h.dim() == 2:
            return h
        if h.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W) or (B,C), got {tuple(h.shape)}")
        B, C, _, _ = h.shape
        return self.spatial_pool(h).view(B, C)

    def forward(self, h_current, h_buffer):
        """
        h_current: (B, C, H, W) - current hidden state
        h_buffer: list of (B, C, H, W) - recent states (detached, max context_len)
        Returns: (B, C, H, W) - temporally-informed hidden state
        """
        B, C, H, W = h_current.shape

        # Pool spatial dims for efficiency: (B, C, H, W) -> (B, C)
        h_pooled = self.pool_state(h_current)

        # Stack recent context (already pooled in buffer for efficiency)
        if len(h_buffer) > 0:
            # Take last context_len frames
            recent = h_buffer[-self.context_len:] if self.context_len > 0 else []
            context_list = [self.pool_state(h) for h in recent]
            context_list.append(h_pooled)
            context = torch.stack(context_list, dim=1)  # (B, T, C)
        else:
            context = h_pooled.unsqueeze(1)  # (B, 1, C)

        T = context.shape[1]

        # QKV projection
        qkv = self.to_qkv(context)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Build causal mask + relative position bias (additive attention mask).
        # Using an explicit mask lets us add rel-pos bias while preserving causality.
        idx = torch.arange(T, device=context.device)
        rel = idx[:, None] - idx[None, :]  # (T, T), i - j
        rel = rel.clamp(-self.context_len, self.context_len) + self.context_len
        bias = self.rel_pos_bias(rel)  # (T, T, heads)
        bias = bias.permute(2, 0, 1)  # (heads, T, T)

        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=context.device, dtype=context.dtype),
            diagonal=1,
        )  # (T, T)

        attn_mask = causal[None, None, :, :] + bias[None, :, :, :]  # (1, heads, T, T)

        # Causal attention (each position attends only to itself and past)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)

        # Take last position (current frame)
        out = self.to_out(attn_out[:, -1])  # (B, C)

        # Broadcast back to spatial dims and add residual
        out = out.view(B, C, 1, 1).expand(B, C, H, W)
        return h_current + out  # Residual connection


class InverseDynamicsModule(nn.Module):
    """
    v6.1: Efficient IDM with Strided Convolutions.
    Predicts cumulative action over k frames from state transitions.
    """
    def __init__(self, hidden_dim: int = 512, action_dim: int = 15, max_span: int = 5):
        super().__init__()

        # Spatial Compressor: 18x32 -> 9x16 -> 5x8
        self.compress = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1),  # -> 9x16
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # -> 5x8
            nn.SiLU(),
            nn.Flatten()
        )

        # 64 channels * 5 * 8 spatial = 2560 features per state
        flattened_dim = 2560
        self.dt_embed = nn.Embedding(max_span + 1, 64)

        # Head
        input_dim = (flattened_dim * 2) + 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim)
        )

    def forward(self, h_start, h_end, dt):
        z_start = self.compress(h_start)  # (B, 2560)
        z_end = self.compress(h_end)      # (B, 2560)
        t_emb = self.dt_embed(dt)         # (B, 64)

        combined = torch.cat([z_start, z_end, t_emb], dim=1)
        return self.mlp(combined)


# ----------------------------
# World model
# ----------------------------

class WorldModelConvFiLM(nn.Module):
    """
    v6.1: Token-level world model with:
    - Separate camera/movement FiLM pathways
    - Lightweight temporal attention for multi-frame comparison
    - 512-dim hidden state for improved capacity
    """
    def __init__(
        self,
        codebook_size: int = 2048,
        embed_dim: int = 256,
        hidden_dim: int = 512,       # v6.1: Up from 384
        n_layers: int = 6,
        H: int = 18,
        W: int = 32,
        action_dim: int = 15,
        idm_max_span: int = 5,
        temporal_context_len: int = 8,  # v6.1: New param
        enable_camera_warp: bool = True,
        max_yaw_warp: int = 8,    # v6.3: Increased for h_prev warping (was 2 for x warping)
        max_pitch_warp: int = 6,  # v6.3: Increased for h_prev warping (was 2 for x warping)
        use_checkpointing: bool = False,
        use_residuals: bool = True,
        zero_init_head: bool = True,
        **kwargs
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.in_proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)

        self.grus = nn.ModuleList([ConvGRUCell(hidden_dim) for _ in range(n_layers)])
        self.film = SeparateActionFiLM(hidden_dim, n_layers)  # v6.1: Separate pathways
        self.temporal_attn = (
            LightweightTemporalAttention(hidden_dim, num_heads=4, context_len=temporal_context_len)
            if temporal_context_len and temporal_context_len > 0
            else None
        )
        self.idm = InverseDynamicsModule(hidden_dim, action_dim, max_span=idm_max_span)

        self.out = nn.Conv2d(hidden_dim, codebook_size, kernel_size=1)

        if zero_init_head:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        self.H, self.W = H, W
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_checkpointing = use_checkpointing
        self.use_residuals = use_residuals
        self.temporal_context_len = temporal_context_len
        self.enable_camera_warp = enable_camera_warp
        self.max_yaw_warp = int(max_yaw_warp)
        self.max_pitch_warp = int(max_pitch_warp)

    def _action_to_pixel_shifts(self, a_t: torch.Tensor):
        """
        Convert discrete yaw/pitch bins to per-sample integer pixel shifts.

        a_t: (B, 15) where yaw is one-hot over 5 bins and pitch is one-hot over 3 bins.
        Returns:
            dx: (B,) int64 horizontal wrap shift (yaw) in [-max_yaw_warp, +max_yaw_warp]
            dy: (B,) int64 vertical padded shift (pitch) in [-max_pitch_warp, +max_pitch_warp]
        """
        if a_t.dim() != 2 or a_t.size(-1) < 8:
            raise ValueError(f"Expected a_t (B,15), got {tuple(a_t.shape)}")

        # Use expected-bin index (works for either one-hot or soft bins).
        yaw_bins = a_t[:, 0:5]
        pitch_bins = a_t[:, 5:8]
        yaw_idx = (yaw_bins * torch.arange(5, device=a_t.device, dtype=a_t.dtype)).sum(dim=1)
        pitch_idx = (pitch_bins * torch.arange(3, device=a_t.device, dtype=a_t.dtype)).sum(dim=1)

        yaw_offset = yaw_idx - 2.0   # center bin at index 2
        pitch_offset = pitch_idx - 1.0  # center bin at index 1

        # Map offsets to pixel shifts.
        # yaw_offset in [-2,2] -> dx in [-max_yaw_warp, max_yaw_warp]
        dx = torch.round(yaw_offset * (self.max_yaw_warp / 2.0)).to(torch.long)
        dy = torch.round(pitch_offset * float(self.max_pitch_warp)).to(torch.long)
        dx = dx.clamp(-self.max_yaw_warp, self.max_yaw_warp)
        dy = dy.clamp(-self.max_pitch_warp, self.max_pitch_warp)
        return dx, dy

    def _apply_yaw_shift(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Per-sample horizontal shift (yaw) for (B,C,H,W) with zero padding at revealed edges.

        Note: This intentionally does NOT wrap around. Unlike panoramic video, Minecraft camera
        turns reveal previously unseen content; the model should learn to inpaint new columns.
        """
        if self.max_yaw_warp <= 0:
            return x
        B, C, H, W = x.shape
        if dx.abs().max().item() == 0:
            return x

        base = torch.arange(W, device=x.device)[None, None, None, :]  # (1,1,1,W)
        # dx > 0 (turn right) should move visible content left, so output[j] samples input[j + dx].
        idx = base + dx.view(B, 1, 1, 1)
        valid = (idx >= 0) & (idx < W)
        idx = idx.clamp(0, W - 1)  # Clamp to avoid out-of-bounds error before masking
        
        idx = idx.expand(B, C, H, W).to(torch.long)
        out = x.gather(dim=3, index=idx)
        
        # Mask out invalid (wrapped) pixels with zeros
        mask = valid.expand(B, C, H, W)
        return out * mask.to(out.dtype)

    def _apply_pitch_shift(self, x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """
        Per-sample vertical shift with zero padding (pitch) for (B,C,H,W).
        """
        if self.max_pitch_warp <= 0:
            return x
        B, C, H, W = x.shape
        if dy.abs().max().item() == 0:
            return x

        y = torch.arange(H, device=x.device)[None, None, :, None]  # (1,1,H,1)
        src_y = y - dy.view(B, 1, 1, 1)  # (B,1,H,1)
        valid = (src_y >= 0) & (src_y < H)
        src_y = src_y.clamp(0, H - 1).to(torch.long)

        src_y_idx = src_y.expand(B, C, H, W)
        out = x.gather(dim=2, index=src_y_idx)
        out = out * valid.to(out.dtype).expand(B, 1, H, 1).expand(B, C, H, W)
        return out

    def _warp_hidden_state(self, h_prev: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """
        v6.3: Warp all layers of hidden state by camera rotation.

        Instead of warping the input embeddings (v6.2), we warp the hidden state.
        This transforms the problem from "predict whole next frame" to "predict residual change".
        The GRU only needs to learn inpainting (filling edges) and dynamics (water, mobs),
        not geometry (camera rotation).

        h_prev: (L, B, C, H, W) - previous hidden states
        dx: (B,) - horizontal shift (yaw), positive = camera turning right
        dy: (B,) - vertical shift (pitch), positive = camera looking up

        Returns: (L, B, C, H, W) - warped hidden states
        """
        L, B, C, H, W = h_prev.shape

        # Reshape for batch processing: (L*B, C, H, W)
        h_flat = h_prev.view(L * B, C, H, W)

        # Replicate dx, dy for each layer
        dx_rep = dx.repeat(L)  # (L*B,)
        dy_rep = dy.repeat(L)  # (L*B,)

        # Apply yaw (horizontal shift with zero padding) and pitch (vertical shift with zero padding)
        h_warped = self._apply_pitch_shift(self._apply_yaw_shift(h_flat, dx_rep), dy_rep)

        return h_warped.view(L, B, C, H, W)

    def _embed_tokens(self, Z):
        x = self.embed(Z)              # (B,H,W,E)
        x = x.permute(0, 3, 1, 2)      # (B,E,H,W)
        x = self.in_proj(x)            # (B,C,H,W)
        return x

    def compute_embeddings(self, Z_seq):
        B, K, H, W = Z_seq.shape
        Z_flat = Z_seq.view(B * K, H, W)
        x_flat = self._embed_tokens(Z_flat)
        _, C, _, _ = x_flat.shape
        return x_flat.view(B, K, C, H, W)

    def compute_film(self, A_seq):
        return self.film(A_seq)

    @torch.no_grad()
    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, self.H, self.W, device=device)

    def step(self, Z_t, a_t, h_prev, x_t=None, gammas_t=None, betas_t=None, temporal_buffer=None):
        """
        Single step forward.

        Args:
            Z_t: (B, H, W) token indices (optional if x_t provided)
            a_t: (B, A) action vector (optional if gammas_t/betas_t provided)
            h_prev: (L, B, C, H, W) previous hidden states
            x_t: (B, C, H, W) pre-computed embeddings (optional)
            gammas_t: (L, B, C, 1, 1) pre-computed FiLM gammas (optional)
            betas_t: (L, B, C, 1, 1) pre-computed FiLM betas (optional)
            temporal_buffer: list of (B, C, H, W) recent hidden states for temporal attention

        Returns:
            logits: (B, codebook_size, H, W)
            new_h: (L, B, C, H, W)
        """
        # Embed inputs
        if x_t is not None:
            x = x_t
        else:
            x = self._embed_tokens(Z_t)

        # FiLM params
        if gammas_t is not None:
            g_all, b_all = gammas_t, betas_t
        else:
            g_all, b_all = self.film(a_t)

        # v6.3: Warp HIDDEN STATE by camera action (instead of warping input as in v6.2).
        # This transforms the problem from "predict whole next frame" to "predict residual".
        # GRU only needs to learn inpainting (filling edges) and dynamics (water, mobs),
        # not geometry (camera rotation).
        if self.enable_camera_warp and a_t is not None:
            dx, dy = self._action_to_pixel_shifts(a_t)
            h_prev = self._warp_hidden_state(h_prev, dx, dy)

        new_h_list = []
        curr = x

        for i, gru in enumerate(self.grus):
            h_p = h_prev[i]
            g = g_all[i]
            b = b_all[i]

            # FiLM Modulation
            modulated = curr * (1.0 + g) + b

            # GRU Update
            h_new = gru(modulated, h_p)

            # Residual Connection (Every 2 layers, skipping 0)
            if self.use_residuals and i > 0 and i % 2 == 0:
                h_new = h_new + curr

            new_h_list.append(h_new)
            curr = h_new

        # v6.1: Apply temporal attention to final layer output
        if self.temporal_attn is not None and temporal_buffer is not None:
            curr = self.temporal_attn(curr, temporal_buffer)

        logits = self.out(curr)
        new_h = torch.stack(new_h_list)
        return logits, new_h

    def forward(self, Z_seq, A_seq, return_all: bool = False):
        """
        Unroll over sequence (used for debugging/validation).
        """
        B, K, H, W = Z_seq.shape
        X_seq = self.compute_embeddings(Z_seq)
        Gammas_seq, Betas_seq = self.compute_film(A_seq)

        h = self.init_state(B)
        logits_list = []
        temporal_buffer = [] if self.temporal_attn is not None and self.temporal_context_len > 0 else None

        for k in range(K):
            # Note: checkpointing doesn't work well with list arguments,
            # so we use same path for both cases
            logits, h = self.step(
                None, A_seq[:, k], h,
                x_t=X_seq[:, k],
                gammas_t=Gammas_seq[:, :, k],
                betas_t=Betas_seq[:, :, k],
                temporal_buffer=temporal_buffer
            )
            logits_list.append(logits)

            # Update temporal buffer (keep last context_len frames, detached)
            if temporal_buffer is not None:
                temporal_buffer.append(self.temporal_attn.pool_state(h[-1].detach()))
                if len(temporal_buffer) > self.temporal_context_len:
                    temporal_buffer.pop(0)

        if return_all:
            return logits_list
        return logits_list[-1]
