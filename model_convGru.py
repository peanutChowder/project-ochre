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
                None, None, h,
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
