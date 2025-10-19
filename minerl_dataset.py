import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class MineRLSequenceDataset(Dataset):
    """
    Loads preprocessed MineRL sequences saved by preprocess.py.
    Each sample is an .npz file with:
        'video'   -> uint8 array [7, 256, 256, 3]
        'actions' -> float32 array [6, 7]
    Returns:
        frames  : float32 tensor [7, 3, 256, 256] in [0,1]
        actions : float32 tensor [6, 7]
    """
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f)
        video = d["video"]        # (7, H, W, 3) uint8
        actions = d["actions"]    # (6, 7) float32

        # convert to torch tensors, normalize frames to [0,1], channels first
        frames = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(actions).float()

        return frames, actions