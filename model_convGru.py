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
        in_ch = channels
        self.z_gate = nn.Conv2d(in_ch + channels, channels, kernel_size, padding=p)
        self.r_gate = nn.Conv2d(in_ch + channels, channels, kernel_size, padding=p)
        self.h_tilde = nn.Conv2d(in_ch + channels, channels, kernel_size, padding=p)

    def forward(self, x, h):
        # x: (B,C,H,W), h: (B,C,H,W)
        z = torch.sigmoid(self.z_gate(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.r_gate(torch.cat([x, h], dim=1)))
        h_tilde = torch.tanh(self.h_tilde(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class ActionFiLM(nn.Module):
    """
    Produces FiLM (gamma, beta) per ConvGRU layer from a low-dim action vector.
    """
    def __init__(self, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.mlps = nn.ModuleList()
        for _ in range(n_layers):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(action_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                )
            )

    def forward(self, a):
        """
        a: (B, A)
        returns list of (gamma, beta), each of shape (B, C, 1, 1)
        """
        outs = []
        for mlp in self.mlps:
            gb = mlp(a)
            g, b = gb.chunk(2, dim=-1)
            # reshape for broadcasting over H,W
            g = g.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)
            outs.append((g, b))
        return outs


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
        embed_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 3,
        H: int = 16,
        W: int = 16,
        action_dim: int = 4,
        use_delta: bool = False,
        zero_init_head: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed = nn.Embedding(codebook_size, embed_dim)
        self.in_proj = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.grus = nn.ModuleList([ConvGRUCell(hidden_dim) for _ in range(n_layers)])
        self.film = ActionFiLM(action_dim, hidden_dim, n_layers)
        self.out = nn.Conv2d(hidden_dim, codebook_size, kernel_size=1)

        # Optional delta mode: learn residual over identity; identity bias via zero init
        self.use_delta = use_delta
        if zero_init_head:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        self.H, self.W = H, W
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _embed_tokens(self, Z):
        # Z: (B,H,W) long
        x = self.embed(Z)              # (B,H,W,E)
        x = x.permute(0, 3, 1, 2)      # (B,E,H,W)
        x = self.in_proj(x)            # (B,C,H,W)
        return x

    @torch.no_grad()
    def init_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return [torch.zeros(batch_size, self.hidden_dim, self.H, self.W, device=device) for _ in range(self.n_layers)]

    def step(self, Z_t, a_t, h_list=None):
        """
        One autoregressive step.
          Z_t: (B,H,W) long indices for current frame
          a_t: (B,A)   actions for transition t -> t+1
          h_list: list of hidden states per layer or None
        Returns:
          logits: (B,codebook_size,H,W)
          new_h: list of hidden states
        """
        x = self._embed_tokens(Z_t)
        if h_list is None:
            h_list = [torch.zeros_like(x) for _ in range(self.n_layers)]
        gammas_betas = self.film(a_t)  # list of (gamma, beta) per layer

        new_h = []
        h = x
        for i, gru in enumerate(self.grus):
            h_prev = h_list[i]
            h = gru(h, h_prev)
            g, b = gammas_betas[i]
            h = h * (1.0 + g) + b
            new_h.append(h)

        logits = self.out(h)  # (B,Kc,H,W)
        return logits, new_h

    def forward(self, Z_seq, A_seq, return_all: bool = False):
        """
        Unroll over a sequence with teacher forcing.

        Z_seq: (B,K,H,W) tokens at times [t0 .. t0+K-1]
        A_seq: (B,K,A)   actions for transitions [t0 .. t0+K-1] (each maps Z_t -> Z_{t+1})

        Returns:
          - logits for next frame after the last input (B,Kc,H,W) if return_all=False
          - list[logits_t] for each step otherwise
        """
        B, K, H, W = Z_seq.shape
        assert (H, W) == (self.H, self.W), f"Got {(H,W)}, expected {(self.H,self.W)}"
        assert A_seq.shape[:2] == (B, K), "A_seq must align with Z_seq along time"

        h_list = None
        logits_list = []
        for k in range(K):
            Z_t = Z_seq[:, k]
            a_t = A_seq[:, k]
            logits, h_list = self.step(Z_t, a_t, h_list)
            logits_list.append(logits)

        if return_all:
            return logits_list
        else:
            return logits_list[-1]
