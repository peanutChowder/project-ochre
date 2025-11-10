#!/usr/bin/env python3
"""
Preprocess MineRL Navigate dataset for autoregressive training.

For each subfolder with recording.mp4 + rendered.npz:
  - Align actions to frames via step→frame mapping (uses metadata duration when available).
  - Downsample in step space to target FPS and resize frames.
  - Encode frames to discrete codes using your trained VQ‑VAE in vq_vae/checkpoints.
  - Save one .npz per trajectory with:
      tokens: [K, H, W] uint16 (K kept frames; H=W≈16 for 64×64 inputs)
      actions: [K−1, 7] float32 (WASD+jump max‑pooled; yaw/pitch averaged)
  - Write out_dir/manifest.json with per‑trajectory lengths to enforce boundaries when sampling sequences.

This avoids mid‑training VQ‑VAE encodes. For n‑step autoregressive loss, sample within a single
trajectory file using manifest lengths so you never cross video boundaries.
"""

import argparse, os, json, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import torch
from vq_vae.vq_vae import Encoder, VectorQuantizerEMA

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

def build_action_matrix(d):
    """Combine action$* arrays into a [T,7] float32 matrix.

    Columns: [forward, left, back, right, jump, yaw, pitch]
    where yaw/pitch are scaled deltas per step (approximately degrees/180).
    """
    camera = d["action$camera"]  # shape (T, 2) [pitch, yaw]
    return np.stack(
        [
            d["action$forward"],
            d["action$left"],
            d["action$back"],
            d["action$right"],
            d["action$jump"],
            np.clip(camera[:, 1] / 180.0, -1, 1),  # yaw
            np.clip(camera[:, 0] / 180.0, -1, 1),  # pitch
        ],
        axis=1,
    ).astype(np.float32)

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


def process_trajectory(
    traj_dir: Path,
    out_dir: Path,
    encoder,
    quantizer,
    device,
    fps_target: int = 8,
    size: int = 64,
    skip_sec: float = 0.0,
    encode_batch: int = 64,
) -> int:
    """
    Process a single trajectory directory.
    Returns 1 if saved successfully.

    Note: During downsampling, movement action booleans are max-pooled within each frame group
    to preserve transient keypresses (e.g., "W" pressed between sampled frames).
    Camera deltas are averaged across the window to preserve smooth movement data.
    Frames are encoded to discrete tokens using TinyVQ-VAE before saving.
    """
    # ---- Load metadata and arrays ----
    meta = {}
    mp = traj_dir / "metadata.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {}

    d = np.load(traj_dir / "rendered.npz", allow_pickle=True)
    per_step_actions = build_action_matrix(d)  # length T
    T = per_step_actions.shape[0]

    # ---- Decode all video frames ----
    frames = [np.asarray(f) for f in iio.imiter(traj_dir / "recording.mp4")]
    F = len(frames)
    if F < 2 or T < 1:
        return 0

    # ---- Optional proportional skip by time ----
    duration_ms = meta.get("duration_ms")
    if duration_ms and duration_ms > 0 and skip_sec > 0.0:
        skip_ms = int(round(skip_sec * 1000))
        skip_steps = int(round(T * (skip_ms / duration_ms)))
        skip_frames = int(round(F * (skip_ms / duration_ms)))
    else:
        print(f"Warning: {traj_dir} has duration {duration_ms}, defaulting to 20FPS")
        skip_steps = 0
        skip_frames = 0

    if skip_steps > 0 or skip_frames > 0:
        per_step_actions = per_step_actions[skip_steps:]
        frames = frames[skip_frames:]
        T = per_step_actions.shape[0]
        F = len(frames)
        if F < 2 or T < 1:
            return 0

    # ---- Map step indices (0..T) to frame indices (0..F-1) ----
    step_to_frame = _safe_linspace_idx(T + 1, F, T + 1)  # length T+1

    # ---- Choose kept state indices in step space using target FPS ----
    if duration_ms and duration_ms > 0:
        eff_ms = max(1, duration_ms - int(round(skip_sec * 1000))) if skip_sec > 0 else duration_ms
        target_states = max(2, int(round((eff_ms / 1000.0) * fps_target)) + 1)
    else:
        # Fallback heuristic if metadata missing
        approx_native = 20.0
        approx_dur_s = F / approx_native
        target_states = max(2, int(round(approx_dur_s * fps_target)) + 1)

    kept_states = np.round(np.linspace(0, T, min(T + 1, target_states))).astype(int)
    kept_states = np.unique(np.append(kept_states, T))
    if kept_states[0] != 0:
        kept_states = np.insert(kept_states, 0, 0)
    if kept_states.size < 2:
        return 0

    # ---- Select frames for kept states and aggregate actions between them ----
    kept_frame_idx = step_to_frame[kept_states]
    kept_frames = [frames[i] for i in kept_frame_idx]

    agg_actions = []
    for i in range(len(kept_states) - 1):
        s0, s1 = kept_states[i], kept_states[i + 1]
        window = per_step_actions[s0:s1]
        if window.shape[0] == 0:
            agg_actions.append(agg_actions[-1] if len(agg_actions) else np.zeros(7, dtype=np.float32))
            continue
        movement = np.max(window[:, 0:5], axis=0)
        camera = np.mean(window[:, 5:7], axis=0)
        agg_actions.append(np.concatenate([movement, camera]).astype(np.float32))
    actions = np.stack(agg_actions, axis=0)

    assert actions.shape[0] == len(kept_frames) - 1, (
        f"Alignment error: actions {actions.shape[0]} vs frames {len(kept_frames)}")

    # ---- Resize frames ----
    T = len(kept_frames)
    resized = np.zeros((T, size, size, 3), dtype=np.uint8)
    for i, img in enumerate(kept_frames):
        im = Image.fromarray(img)
        im = im.resize((size, size), Image.BICUBIC)
        resized[i] = np.asarray(im, dtype=np.uint8)

    # ---- Encode frames to discrete tokens using trained VQ-VAE ----
    tokens_list = []
    encoder.eval(); quantizer.eval()
    with torch.no_grad():
        for i in range(0, T, encode_batch):
            chunk = torch.from_numpy(resized[i:i+encode_batch]).permute(0,3,1,2).float() / 255.0
            chunk = chunk.to(device)
            z_e = encoder(chunk)
            _, _, _, enc_idx = quantizer(z_e)
            H, W = z_e.shape[2], z_e.shape[3]
            enc_idx = enc_idx.view(-1, H, W)
            tokens_list.append(enc_idx.cpu().to(torch.int32))
    tokens = torch.cat(tokens_list, dim=0).numpy().astype(np.uint16)

    # ---- Save full trajectory tokens and actions ----
    out_name = f"{traj_dir.name}.npz"
    # tokens.shape[0] == actions.shape[0] + 1
    np.savez_compressed(
        out_dir / out_name,
        tokens=tokens,
        actions=actions,
    )
    print(f"Saved tokens and actions for trajectory {traj_dir.name}")
    return 1

def main(parent_dir: str,
         out_dir: str,
         fps_target: int = 8,
         size: int = 64,
         skip_sec: float = 0.0,
         vqvae_ckpt: str = "vq_vae/checkpoints/vqvae_epoch_10.pt",
         encode_batch: int = 64):
    parent = Path(parent_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, quantizer = load_vqvae(vqvae_ckpt, device)

    trajs = [p for p in parent.iterdir()
             if p.is_dir() and (p / "rendered.npz").exists()
                             and (p / "recording.mp4").exists()]

    total = 0
    for traj in tqdm(trajs, desc="Processing trajectories"):
        total += process_trajectory(
            traj,
            out,
            encoder=encoder,
            quantizer=quantizer,
            device=device,
            fps_target=fps_target,
            size=size,
            skip_sec=skip_sec,
            encode_batch=encode_batch,
        )
    print(f"\nProcessed {len(trajs)} trajectories")
    print(f"Saved {total} sequences to {out}")
    # Emit a manifest summarizing per-trajectory sequence lengths
    manifest = []
    for traj in trajs:
        out_name = f"{traj.name}.npz"
        op = out / out_name
        if op.exists():
            try:
                z = np.load(op)
                manifest.append({"file": out_name, "length": int(z["tokens"].shape[0])})
            except Exception:
                pass
    (out / "manifest.json").write_text(json.dumps({"sequences": manifest}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent_dir", required=True,
                   help="Path to MineRLNavigate-v0 parent directory")
    p.add_argument("--out_dir", required=True,
                   help="Directory to save processed sequences")
    p.add_argument("--fps_target", type=int, default=8,
                   help="Target FPS after downsampling (default 8)")
    p.add_argument("--size", type=int, default=64,
                   help="Output frame size (default 64)")
    p.add_argument("--skip_sec", type=float, default=0.0,
                   help="Skip this many seconds at start (proportional)")
    p.add_argument("--vqvae_ckpt", type=str, default="vq_vae/checkpoints/vqvae_epoch_10.pt",
                   help="Path to trained VQ-VAE checkpoint for encoding frames")
    p.add_argument("--encode_batch", type=int, default=64,
                   help="Batch size for VQ-VAE encoding")
    args = p.parse_args()
    main(**vars(args))
