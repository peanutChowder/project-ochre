import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ----------------------------
# Core building blocks
# ----------------------------

class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        # Fused z and r gates: Input (in_ch + channels) -> Output (2 * channels)
        # in_ch is assumed to be equal to channels here based on usage
        self.zr_gate = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=p)
        self.h_tilde_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=p)

    def forward(self, x, h):
        # x: (B,C,H,W), h: (B,C,H,W)
        
        # 1. Compute z and r gates together
        combined_zr = torch.cat([x, h], dim=1)
        zr = self.zr_gate(combined_zr)
        z, r = zr.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # 2. Compute candidate hidden state
        combined_h = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.h_tilde_gate(combined_h))

        # 3. Update hidden state
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class ActionFiLM(nn.Module):
    """
    Produces FiLM (gamma, beta) per ConvGRU layer from a low-dim action vector.

    v4.7.0: Increased internal MLP hidden dimension to 512 (was same as hidden_dim=256)
    to strengthen action conditioning. Camera rotations require complex geometric mappings.

    v5.0: Scale internal dimension with both action_dim and hidden_dim for 15D actions.
    """
    def __init__(self, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.mlps = nn.ModuleList()
        # v5.0: Scale with both action_dim and hidden_dim
        # For action_dim=15, hidden_dim=640: 640 * 2 = 1280
        film_internal_dim = max(1024, hidden_dim * 2)
        for _ in range(n_layers):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(action_dim, film_internal_dim),
                    nn.SiLU(),
                    nn.Linear(film_internal_dim, 2 * hidden_dim),
                )
            )

    def forward(self, a):
        """
        a: (B, A) or (B, T, A)
        returns (gammas, betas) as stacked tensors.
        gammas, betas: (L, B, C, 1, 1) or (L, B, T, C, 1, 1)
        """
        is_seq = a.dim() == 3
        gammas = []
        betas = []
        for mlp in self.mlps:
            gb = mlp(a) # (B, 2C) or (B, T, 2C)
            g, b = gb.chunk(2, dim=-1)
            
            if is_seq:
                # (B, T, C) -> (B, T, C, 1, 1)
                g = g.unsqueeze(-1).unsqueeze(-1)
                b = b.unsqueeze(-1).unsqueeze(-1)
            else:
                # (B, C) -> (B, C, 1, 1)
                g = g.unsqueeze(-1).unsqueeze(-1)
                b = b.unsqueeze(-1).unsqueeze(-1)
            
            gammas.append(g)
            betas.append(b)

        # v4.11.0: Hard Clamp to prevent 1-frame flashes
        # Gamma + 1.0 ranges 0.5 to 2.0. Values > 5.0 cause washout artifacts.
        gammas = torch.stack(gammas)
        betas = torch.stack(betas)

        gammas = torch.clamp(gammas, -5.0, 5.0)
        betas = torch.clamp(betas, -5.0, 5.0)

        return gammas, betas


class InverseDynamicsModule(nn.Module):
    """
    v4.11.0: Variable-Span IDM (The "Time Telescope")
    Predicts cumulative action between h_t and h_{t+k} where k ∈ [1, max_span].
    Includes Time-Delta embedding to tell the network the span length.

    Theory: Movement velocity is visible over 3-5 frames (player displacement),
    not 1 frame. Multi-step action prediction forces model to encode movement
    momentum and temporal dynamics.

    v5.0: AGGRESSIVE spatial pooling (18×32 → 4×8) to prevent parameter explosion
    with larger hidden_dim. 3-layer MLP with dropout for better regularization.
    """
    def __init__(self, hidden_dim: int = 640, action_dim: int = 15, max_span: int = 5):
        super().__init__()

        # v5.0: AGGRESSIVE pooling 18×32 → 4×8 (16× spatial reduction)
        # Maintains coarse spatial awareness (left/right, top/bottom quadrants)
        # Prevents parameter explosion: 640×4×8 = 20,480 (vs 640×9×16 = 92,160)
        self.pool = nn.AdaptiveAvgPool2d((4, 8))

        # v4.11.0: Time-Delta Embedding
        # Encodes scalar 'k' (span length) into a vector
        self.dt_embed = nn.Embedding(max_span + 1, 64)

        # Input: (Pooled_State * 2) + Time_Embedding
        pooled_size = hidden_dim * 4 * 8  # 640 * 32 = 20,480 per state
        input_dim = (pooled_size * 2) + 64  # 40,960 + 64 = 41,024

        # v5.0: 3-layer MLP with dropout for better action abstraction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, h_start, h_end, dt):
        """
        Args:
            h_start: (B, hidden_dim, H, W) hidden state at time t-k
            h_end: (B, hidden_dim, H, W) hidden state at time t
            dt: (B,) Integer tensor of time deltas (span lengths, e.g., [1, 3, 5...])

        Returns:
            action_pred: (B, action_dim) predicted cumulative action over span
        """
        # 1. Pool and Flatten States
        z_start = self.pool(h_start).flatten(1)  # (B, 36864)
        z_end = self.pool(h_end).flatten(1)      # (B, 36864)

        # 2. Embed Time Delta
        t_emb = self.dt_embed(dt)  # (B, 64)

        # 3. Concatenate and Predict
        # [Start State, End State, Time Gap] -> Cumulative Action
        combined = torch.cat([z_start, z_end, t_emb], dim=1)  # (B, 73792)
        return self.mlp(combined)  # (B, action_dim)


# ----------------------------
# World model: token space ConvGRU + FiLM
# ----------------------------

class WorldModelConvFiLM(nn.Module):
    """
    Token-level world model operating on VQ-VAE indices.

    Inputs:
      - Z_seq: (B, K, H, W)  integer token indices (long)
      - A_seq: (B, K, A)     continuous actions in [-1,1], A=4: [yaw, pitch, move_x, move_z]

    Outputs:
      - If return_all=False: logits for next step after last input  (B, Kc, H, W) with Kc=codebook_size
      - If return_all=True:  list of logits per step

    Use step() for single-step autoregressive rollouts.
    """
    def __init__(
        self,
        codebook_size: int = 2048,
        embed_dim: int = 320,        # v5.0: Was 256
        hidden_dim: int = 640,       # v5.0: Was 256
        n_layers: int = 6,           # v5.0: Was 3
        H: int = 16,
        W: int = 16,
        action_dim: int = 15,        # v5.0: Was 4 (now 15D discrete)
        idm_max_span: int = 5,
        use_delta: bool = False,
        zero_init_head: bool = True,
        use_checkpointing: bool = False,
        use_residuals: bool = True,  # v5.0: NEW - residual connections every 2 layers
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.in_proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.grus = nn.ModuleList([ConvGRUCell(hidden_dim) for _ in range(n_layers)])
        self.film = ActionFiLM(action_dim, hidden_dim, n_layers)
        # v4.10.0: Inverse Dynamics Module for action conditioning
        self.idm = InverseDynamicsModule(hidden_dim, action_dim, max_span=idm_max_span)
        self.out = nn.Conv2d(hidden_dim, codebook_size, kernel_size=1)

        # Optional delta mode: learn residual over identity; identity bias via zero init
        self.use_delta = use_delta
        if zero_init_head:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        self.H, self.W = H, W
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_checkpointing = use_checkpointing
        self.use_residuals = use_residuals  # v5.0

    def _embed_tokens(self, Z):
        """
        Public wrapper for token embedding (needed for scheduled sampling).
        Z: (B,H,W) long
        Returns: (B,C,H,W)
        """
        x = self.embed(Z)              # (B,H,W,E)
        x = x.permute(0, 3, 1, 2)      # (B,E,H,W)
        x = self.in_proj(x)            # (B,C,H,W)
        return x

    def compute_embeddings(self, Z_seq):
        """
        Pre-compute embeddings for a whole sequence.
        Z_seq: (B, K, H, W)
        Returns: (B, K, C, H, W)
        """
        B, K, H, W = Z_seq.shape
        # Flatten time into batch for efficient computation
        Z_flat = Z_seq.view(B * K, H, W)
        x_flat = self._embed_tokens(Z_flat) # (B*K, C, H, W)
        _, C, _, _ = x_flat.shape
        return x_flat.view(B, K, C, H, W)

    def compute_film(self, A_seq):
        """
        Pre-compute FiLM parameters for a whole sequence.
        A_seq: (B, K, A)
        Returns: (Gammas, Betas) stacked tensors (L, B, K, C, 1, 1)
        """
        return self.film(A_seq)

    @torch.no_grad()
    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # Return stacked tensor (L, B, C, H, W)
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, self.H, self.W, device=device)

    def step(self, Z_t, a_t, h_prev, x_t=None, gammas_t=None, betas_t=None):
        """
        One autoregressive step.
          Z_t: (B,H,W) long indices for current frame (optional if x_t provided)
          a_t: (B,A)   actions for transition t -> t+1 (optional if gammas/betas provided)
          h_prev: (L, B, C, H, W) hidden states
          x_t: (B,C,H,W) pre-computed embedding (optional)
          gammas_t, betas_t: (L, B, C, 1, 1) (optional)
        Returns:
          logits: (B,codebook_size,H,W)
          new_h: (L, B, C, H, W)
        """
        # Use pre-computed x_t if available, otherwise compute from Z_t
        if x_t is not None:
            x = x_t
        else:
            x = self._embed_tokens(Z_t)

        if h_prev is None:
            # This case handles initialization if None passed, but signature expects tensor now.
            # We assume h_prev is provided or we construct it. 
            # But better to assume caller provides it.
             pass 
        
        # Use pre-computed film if available
        if gammas_t is not None and betas_t is not None:
             pass
        else:
            gammas_t, betas_t = self.film(a_t)

        new_h_list = []
        # 'current_layer_input' here is the input feature map for the current ConvGRU layer.
        # For the first layer, it's 'x' (the embedded visual token).
        # For subsequent layers, it's the output of the previous ConvGRU layer.
        current_layer_input = x 

        for i, gru in enumerate(self.grus):
            h_p = h_prev[i] # This is the hidden state from the *previous time step* for *this* ConvGRU layer.
            
            # Get FiLM parameters for this layer and current time step
            g = gammas_t[i] # (B, C, 1, 1)
            b = betas_t[i]  # (B, C, 1, 1)

            # Apply FiLM modulation to the *input* of the current ConvGRU layer.
            # This ensures the action influences the GRU's gates and candidate state calculation.
            modulated_input_to_gru = current_layer_input * (1.0 + g) + b

            # Compute the new hidden state for *this* GRU layer at *this* time step.
            # This is the recurrent state that will be passed to the *next time step*.
            h_new_recurrent_state = gru(modulated_input_to_gru, h_p)

            # v5.0: Residual connections every 2 layers (for gradient flow in deeper 6-layer stack)
            if self.use_residuals and i > 0 and i % 2 == 0:
                h_new_recurrent_state = h_new_recurrent_state + current_layer_input

            # Store this new recurrent state.
            new_h_list.append(h_new_recurrent_state)
            
            # The output of this GRU layer (`h_new_recurrent_state`) becomes the
            # `current_layer_input` for the *next GRU layer* in the stack (same time step).
            current_layer_input = h_new_recurrent_state
            

        logits = self.out(current_layer_input)  # (B,Kc,H,W)
        new_h = torch.stack(new_h_list)
        return logits, new_h

    def forward(self, Z_seq, A_seq, return_all: bool = False):
        """
        Unroll over a sequence with teacher forcing.
        """
        B, K, H, W = Z_seq.shape
        assert (H, W) == (self.H, self.W), f"Got {(H,W)}, expected {(self.H,self.W)}"
        assert A_seq.shape[:2] == (B, K), "A_seq must align with Z_seq along time"

        # Pre-compute for efficiency in unroll
        X_seq = self.compute_embeddings(Z_seq)
        Gammas_seq, Betas_seq = self.compute_film(A_seq)
        
        h = self.init_state(B)
        logits_list = []
        for k in range(K):
            # Extract step data
            x_t = X_seq[:, k]
            # (L, B, K, C, 1, 1) -> (L, B, C, 1, 1)
            g_t = Gammas_seq[:, :, k]
            b_t = Betas_seq[:, :, k]
            
            if self.use_checkpointing:
                # checkpoint requires tensors. h is tensor, x_t is tensor, g_t, b_t are tensors.
                # Z_t=None, a_t=None.
                # We pass dummy None for Z_t, a_t if we rely on positional args
                logits, h = checkpoint(self.step, torch.tensor(0), torch.tensor(0), h, x_t, g_t, b_t, use_reentrant=False)
                # Note: We passed dummy tensors because passing None to checkpoint can be tricky with some versions
                # But inside step we check if x_t is Not None.
                # Let's ensure step handles dummy inputs gracefully.
                # step signature: (Z_t, a_t, h_prev, x_t, gammas_t, betas_t)
                # We passed: (tensor(0), tensor(0), h, x_t, g_t, b_t)
                # inside step: x_t is not None, so Z_t ignored. gammas_t is not None, so a_t ignored.
                # Perfect.
            else:
                logits, h = self.step(None, None, h, x_t, g_t, b_t)
            
            logits_list.append(logits)

        if return_all:
            return logits_list
        else:
            return logits_list[-1]
