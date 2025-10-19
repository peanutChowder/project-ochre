import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    """Encodes 7-dim action vectors -> transformer hidden size."""
    def __init__(self, action_dim=7, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, a):
        return self.net(a)  # [B, K, hidden_dim]

class TemporalTransformer(nn.Module):
    """
    Predict next latent given K latent frames and actions.
    latents: [B, K, C, H, W] -> flattened to [B, K, token_dim]
    actions: [B, K, 7]
    """
    def __init__(self, latent_channels=4, H=32, W=32,
                 hidden_dim=512, n_heads=8, n_layers=6):
        super().__init__()
        token_dim = latent_channels * H * W
        self.latent_embed = nn.Linear(token_dim, hidden_dim)
        self.action_enc = ActionEncoder(7, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out = nn.Linear(hidden_dim, token_dim)

        self.latent_channels, self.H, self.W = latent_channels, H, W

    def forward(self, latents, actions):
        B, K, C, H, W = latents.shape
        lat_flat = latents.view(B, K, -1)
        x = self.latent_embed(lat_flat) + self.action_enc(actions)
        x = self.transformer(x)[:, -1]  # take final token
        pred = self.out(x).view(B, C, H, W)
        return pred