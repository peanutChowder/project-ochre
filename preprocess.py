#!/usr/bin/env python3
"""
Preprocess GameFactory dataset into VQ‑VAE token + action sequences for autoregressive training.

Layout (data_2003):
- parent_dir/metadata/*.json  (per‑frame actions)
- parent_dir/video/*.mp4      (matching basename)
- parent_dir/annotation.csv   (optional, not used here)

Pipeline:
- Align actions to frames via simple linspace mapping.
- Optionally downsample in step space to a target FPS (default 16).
- Resize frames to the VQ‑VAE training resolution (IMAGE_WIDTH x IMAGE_HEIGHT).
- Encode frames to discrete codes using the trained VQ‑VAE.
- Save one .npz per video with:
    tokens: [K, H, W] uint16
    actions: [K−1, 5] float32 = [yaw, pitch, move_x, move_z, jump]
- Maintain manifest.json with per‑trajectory lengths.

This avoids mid‑training VQ‑VAE encodes. For n‑step autoregressive loss, sample within a single
trajectory file using manifest lengths so you never cross video boundaries.
"""

import argparse, os, json, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import torch
from vq_vae.vq_vae import Encoder, VectorQuantizerEMA, IMAGE_HEIGHT, IMAGE_WIDTH
from action_encoding import encode_action_v5_np

PROGRESS_FILENAME = "progress.json"

def _safe_linspace_idx(count_from: int, count_to: int, n: int) -> np.ndarray:
    """Integer index mapping by rounding linspace.

    Example: map (T+1) state steps to F video frames via round(linspace(0,F-1,T+1)).
    Returns array of length n with values in [0, count_to-1].
    """
    if n <= 1 or count_to <= 0:
        return np.array([0], dtype=int)
    xs = np.linspace(0, max(count_to - 1, 0), n)
    idx = np.round(xs).astype(int)
    return np.clip(idx, 0, max(count_to - 1, 0))


def build_action_matrix_gamefactory(actions_dict: dict, angle_scale: float = 0.3) -> np.ndarray:
    """
    Build [T,5] float32 matrix from GameFactory metadata actions.

    Expected per-step keys:
      - ws: 0/1/2 → none / W (forward) / S (backward)
      - ad: 0/1/2 → none / A (left) / D (right)
      - yaw_delta, pitch_delta: float deltas
      - scs: 0/1/2/3 → none / jump (Space) / sneak (Shift) / sprint (Ctrl)

    Mapping:
      move_z: +1 = forward (W), -1 = backward (S), 0 = none
      move_x: -1 = left (A), +1 = right (D), 0 = none
      yaw/pitch are scaled deltas, roughly normalized to [-1,1].
      jump is a binary flag derived from scs (1 when Space/jump is active).
    """
    idxs = sorted(actions_dict.keys(), key=lambda k: int(k))
    T = len(idxs)
    yaw = np.zeros(T, dtype=np.float32)
    pitch = np.zeros(T, dtype=np.float32)
    move_x = np.zeros(T, dtype=np.float32)
    move_z = np.zeros(T, dtype=np.float32)
    jump = np.zeros(T, dtype=np.float32)
    for i, k in enumerate(idxs):
        a = actions_dict[k]
        ws = int(a.get("ws", 0))
        ad = int(a.get("ad", 0))
        scs = int(a.get("scs", 0))
        yaw_delta = float(a.get("yaw_delta", 0.0))
        pitch_delta = float(a.get("pitch_delta", 0.0))
        # Forward/back: 1=W (forward) → +1, 2=S (backward) → -1
        if ws == 1:
            move_z[i] = 1.0
        elif ws == 2:
            move_z[i] = -1.0
        else:
            move_z[i] = 0.0

        # Strafe: 1=A (left) → -1, 2=D (right) → +1
        if ad == 1:
            move_x[i] = -1.0
        elif ad == 2:
            move_x[i] = 1.0
        else:
            move_x[i] = 0.0
        yaw[i] = yaw_delta / max(angle_scale, 1e-6)
        pitch[i] = pitch_delta / max(angle_scale, 1e-6)
        # scs == 1 corresponds to Space/jump; other values are ignored for now.
        jump[i] = 1.0 if scs == 1 else 0.0
    yaw = np.clip(yaw, -1.0, 1.0)
    pitch = np.clip(pitch, -1.0, 1.0)
    return np.stack([yaw, pitch, move_x, move_z, jump], axis=1).astype(np.float32)


def build_action_matrix_gamefactory_v5(actions_dict: dict, angle_scale: float = 0.3) -> np.ndarray:
    """
    v5.0: Build [T, 15] float32 matrix with discrete action encoding.

    Format: [yaw(5), pitch(3), W, A, S, D, jump, sprint, sneak]

    Yaw bins (5D one-hot):
      Bin 0: [-1.0, -0.5] = Hard Left
      Bin 1: (-0.5, -0.1] = Left
      Bin 2: (-0.1, 0.1]  = Center
      Bin 3: (0.1, 0.5]   = Right
      Bin 4: (0.5, 1.0]   = Hard Right

    Pitch bins (3D one-hot):
      Bin 0: [-1.0, -0.2] = Down
      Bin 1: (-0.2, 0.2]  = Level
      Bin 2: (0.2, 1.0]   = Up

    WASD (4D multi-hot): Can have multiple 1s for diagonal movement
      Dim 8:  W (forward)
      Dim 9:  A (left)
      Dim 10: S (backward)
      Dim 11: D (right)

    Jump/Sprint/Sneak (3D binary):
      Dim 12: Jump (Space, scs=1)
      Dim 13: Sprint (Ctrl, scs=3)
      Dim 14: Sneak (Shift, scs=2)

    Expected per-step keys:
      - ws: 0/1/2 → none / W (forward) / S (backward)
      - ad: 0/1/2 → none / A (left) / D (right)
      - yaw_delta, pitch_delta: float deltas
      - scs: 0/1/2/3 → none / jump (Space) / sneak (Shift) / sprint (Ctrl)
    """
    T = len(actions_dict)
    action_matrix = np.zeros((T, 15), dtype=np.float32)

    for i, k in enumerate(sorted(actions_dict.keys(), key=int)):
        a = actions_dict[k]
        yaw_raw = float(a.get("yaw_delta", 0.0)) / max(angle_scale, 1e-6)
        pitch_raw = float(a.get("pitch_delta", 0.0)) / max(angle_scale, 1e-6)

        ws = int(a.get("ws", 0))
        ad = int(a.get("ad", 0))
        scs = int(a.get("scs", 0))

        action_matrix[i] = encode_action_v5_np(
            yaw_raw=yaw_raw,
            pitch_raw=pitch_raw,
            w=(ws == 1),
            s=(ws == 2),
            a=(ad == 1),
            d=(ad == 2),
            jump=(scs == 1),
            sneak=(scs == 2),
            sprint=(scs == 3),
        )

    return action_matrix


def build_action_matrix_gamefactory_v412(actions_dict: dict, angle_scale: float = 0.3) -> np.ndarray:
    """Back-compat alias for v5.0 15D discrete action encoding."""
    return build_action_matrix_gamefactory_v5(actions_dict, angle_scale=angle_scale)


def load_vqvae(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    emb_dim = 256
    num_emb = 2048
    if isinstance(ckpt, dict):
        if "quantizer" in ckpt and isinstance(ckpt["quantizer"], dict):
            emb = ckpt["quantizer"].get("embedding")
            if isinstance(emb, torch.Tensor) and emb.ndim == 2:
                emb_dim, num_emb = int(emb.shape[0]), int(emb.shape[1])
        cfg = ckpt.get("config") or {}
        emb_dim = int(cfg.get("embedding_dim", emb_dim))
        num_emb = int(cfg.get("num_embeddings", num_emb))
    enc = Encoder(3, embedding_dim=emb_dim).to(device)
    quant = VectorQuantizerEMA(embedding_dim=emb_dim, num_embeddings=num_emb).to(device)
    if isinstance(ckpt, dict) and all(k in ckpt for k in ["encoder", "quantizer"]):
        enc.load_state_dict(ckpt["encoder"]) 
        quant.load_state_dict(ckpt["quantizer"]) 
    else:
        tmp = torch.nn.Module(); tmp.encoder = enc; tmp.quantizer = quant
        tmp.load_state_dict(ckpt, strict=False)
    enc.eval(); quant.eval()
    return enc, quant

def _load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {"processed": []}
    try:
        with progress_path.open("r") as f:
            return json.load(f)
    except Exception:
        return {"processed": []}


def _save_progress(progress_path: Path, progress: dict) -> None:
    tmp = progress_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(progress_path)


def _update_manifest(out_dir: Path) -> None:
    manifest = []
    for npz_path in sorted(out_dir.glob("*.npz")):
        if npz_path.name == "manifest.json":
            continue
        try:
            z = np.load(npz_path)
            manifest.append({"file": npz_path.name, "length": int(z["tokens"].shape[0])})
        except Exception:
            continue
    (out_dir / "manifest.json").write_text(json.dumps({"sequences": manifest}, indent=2))


def main(parent_dir: str,
         out_dir: str,
         fps_target: int = 16,
         vqvae_ckpt: str = "vq_vae/checkpoints/vqvae_epoch_10.pt",
         encode_batch: int = 64,
         angle_scale: float = 0.3):
    """
    parent_dir: GameFactory data_2003 directory containing metadata/ and video/.
    fps_target: target FPS after downsampling in step space (default 16).
    """
    parent = Path(parent_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, quantizer = load_vqvae(vqvae_ckpt, device)

    progress_path = out / PROGRESS_FILENAME
    progress = _load_progress(progress_path)
    processed_set = set(progress.get("processed", []))

    meta_dir = parent / "metadata"
    video_dir = parent / "video"
    meta_files = sorted(meta_dir.glob("*.json"))
    total = 0
    for meta_path in tqdm(meta_files, desc="Processing GameFactory videos"):
        stem = meta_path.stem
        if stem in processed_set:
            continue
        video_path = video_dir / (stem + ".mp4")
        if not video_path.exists():
            continue
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
        except Exception:
            continue

        actions_dict = meta.get("actions", {})
        if not actions_dict:
            continue
        # v5.0: Use 15D discrete action encoding
        per_step_actions = build_action_matrix_gamefactory_v5(actions_dict, angle_scale=angle_scale)
        T = per_step_actions.shape[0]

        try:
            frames = [np.asarray(f) for f in iio.imiter(video_path)]
        except Exception:
            continue
        F = len(frames)
        if F < 2 or T < 1:
            continue

        step_to_frame = _safe_linspace_idx(T + 1, F, T + 1)

        approx_native = 16.0
        approx_dur_s = T / approx_native
        if fps_target > 0:
            target_states = max(2, int(round(approx_dur_s * fps_target)) + 1)
        else:
            target_states = T + 1
        kept_states = np.round(np.linspace(0, T, min(T + 1, target_states))).astype(int)
        kept_states = np.unique(np.append(kept_states, T))
        if kept_states[0] != 0:
            kept_states = np.insert(kept_states, 0, 0)
        if kept_states.size < 2:
            continue

        kept_frame_idx = step_to_frame[kept_states]
        kept_frames = [frames[i] for i in kept_frame_idx]

        agg_actions = []
        for i in range(len(kept_states) - 1):
            s0, s1 = kept_states[i], kept_states[i + 1]
            window = per_step_actions[s0:s1]
            if window.shape[0] == 0:
                agg_actions.append(agg_actions[-1] if len(agg_actions) else np.zeros(5, dtype=np.float32))
                continue
            pooled = np.mean(window, axis=0)
            agg_actions.append(pooled.astype(np.float32))
        actions = np.stack(agg_actions, axis=0)

        if actions.shape[0] != len(kept_frames) - 1:
            continue

        T_kept = len(kept_frames)
        resized = np.zeros((T_kept, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        for i, img in enumerate(kept_frames):
            im = Image.fromarray(img)
            im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BICUBIC)
            resized[i] = np.asarray(im, dtype=np.uint8)

        tokens_list = []
        encoder.eval(); quantizer.eval()
        with torch.no_grad():
            for i in range(0, T_kept, encode_batch):
                chunk = torch.from_numpy(resized[i:i+encode_batch]).permute(0,3,1,2).float() / 255.0
                chunk = chunk.to(device)
                z_e = encoder(chunk)
                _, _, _, enc_idx = quantizer(z_e)
                H, W = z_e.shape[2], z_e.shape[3]
                enc_idx = enc_idx.view(-1, H, W)
                tokens_list.append(enc_idx.cpu().to(torch.int32))
        tokens = torch.cat(tokens_list, dim=0).numpy().astype(np.uint16)

        out_name = f"{stem}.npz"
        np.savez_compressed(out / out_name, tokens=tokens, actions=actions)
        total += 1
        processed_set.add(stem)
        progress["processed"] = sorted(processed_set)
        _save_progress(progress_path, progress)

    print(f"\nProcessed {total} GameFactory videos into {out}")

    _update_manifest(out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent_dir", required=True,
                   help="Path to GameFactory data_2003 directory (with metadata/ and video/)")
    p.add_argument("--out_dir", required=True,
                   help="Directory to save processed sequences")
    p.add_argument("--fps_target", type=int, default=16,
                   help="Target FPS after downsampling (default 16)")
    p.add_argument("--vqvae_ckpt", type=str, default="vq_vae/checkpoints/vqvae_epoch_10.pt",
                   help="Path to trained VQ-VAE checkpoint for encoding frames")
    p.add_argument("--encode_batch", type=int, default=64,
                   help="Batch size for VQ-VAE encoding")
    p.add_argument("--angle_scale", type=float, default=1.0,
                   help="Scale factor for GameFactory yaw/pitch deltas before clipping to [-1,1]")
    args = p.parse_args()
    main(**vars(args))
