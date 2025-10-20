import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class MineRLFrameTripletDataset(Dataset):
    """
    Loads full MineRL trajectories saved as .npz files.
    Each .npz file contains:
        'video'   -> uint8 array [T, 256, 256, 3]
        'actions' -> float32 array [T, 7]
    Each __getitem__ returns:
        (frame_t, action_t, frame_next, frame_next2)
        frame_t:    float32 tensor [3, 256, 256] in [0,1]
        action_t:   float32 tensor [7]
        frame_next: float32 tensor [3, 256, 256] in [0,1]
        frame_next2:float32 tensor [3, 256, 256] in [0,1]
    Deterministic: indexes every valid (t, t+1, t+2) triplet once per epoch.
    """

    def __init__(self, root_dir, min_length=3):
        self.files = sorted(list(Path(root_dir).glob("*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {root_dir}")

        self.triplets = []
        for i, f in enumerate(self.files):
            try:
                with np.load(f) as d:
                    video = d["video"]
                    actions = d["actions"]
                    if (
                        isinstance(video, np.ndarray)
                        and isinstance(actions, np.ndarray)
                        and video.ndim == 4
                        and video.shape[1:4] == (256, 256, 3)
                        and actions.shape[1] == 7
                        and video.shape[0] >= min_length
                        and actions.shape[0] == video.shape[0]
                    ):
                        n_frames = video.shape[0]
                        for t in range(n_frames - 2):
                            self.triplets.append((i, t))
            except Exception:
                continue

        if not self.triplets:
            raise RuntimeError("No valid frame triplets found.")

        print(f" Indexed {len(self.triplets)} total frame triplets from {len(self.files)} videos.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        file_idx, t = self.triplets[idx]
        d = np.load(self.files[file_idx])
        video = d["video"]      # (T, 256, 256, 3)
        actions = d["actions"]  # (T, 7)

        frame_t    = torch.from_numpy(video[t]).permute(2, 0, 1).float() / 255.0
        frame_next = torch.from_numpy(video[t + 1]).permute(2, 0, 1).float() / 255.0
        frame_next2= torch.from_numpy(video[t + 2]).permute(2, 0, 1).float() / 255.0
        action_t   = torch.from_numpy(actions[t]).float()

        return frame_t, action_t, frame_next, frame_next2