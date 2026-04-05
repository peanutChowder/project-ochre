#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.analyze_checkpoint import get_test_actions, load_checkpoint
from diagnostics.visual_quality_eval import _edge_map, compute_metrics, prime_temporal_buffer

CANONICAL_CONTEXTS = [
    str(REPO_ROOT / "preprocessedv5/seed_1000_part_1000.npz"),
    str(REPO_ROOT / "preprocessedv5/seed_1001_part_1001.npz"),
    str(REPO_ROOT / "preprocessedv5/seed_1002_part_1002.npz"),
    str(REPO_ROOT / "preprocessedv5/seed_1728_part_1728.npz"),
]

DEFAULT_ROLLOUT = {"topk": 50, "temperature": 1.0, "start_idx": 8, "rollout_steps": 20}
CONTACT_SHEET_COLS = [0, 1, 2, 4, 8, 12, 16, 19]


def select_device(preferred: str = "mps") -> torch.device:
    if torch.cuda.is_available() and preferred == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available() and preferred == "mps":
        return torch.device("mps")
    if torch.cuda.is_available() and preferred not in {"mps", "cpu"}:
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_context_paths(context_npz: list[str] | None, data_dir: str | None = None) -> list[str]:
    if context_npz:
        return context_npz
    base = Path(data_dir or (REPO_ROOT / "preprocessedv5"))
    return [str(base / Path(name).name) for name in CANONICAL_CONTEXTS]


def load_model_bundle(checkpoint: str, vqvae_ckpt: str, device: torch.device):
    return load_checkpoint(checkpoint, vqvae_ckpt, device)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def edge_to_pil(edge: torch.Tensor) -> Image.Image:
    arr = edge.detach().cpu().squeeze().numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


def make_token_agreement_heatmap(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor) -> torch.Tensor:
    agreement = (pred_tokens == gt_tokens).float().unsqueeze(1)
    h, w = gt_tokens.shape[-2:]
    agreement_up = F.interpolate(agreement, size=(h * 4, w * 4), mode="nearest")
    heatmap = torch.zeros(agreement_up.shape[0], 3, agreement_up.shape[2], agreement_up.shape[3], device=pred_tokens.device)
    heatmap[:, 0] = 1.0 - agreement_up[:, 0]
    heatmap[:, 1] = agreement_up[:, 0]
    return heatmap


def token_agreement_to_pil(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor) -> Image.Image:
    correct = (pred_tokens == gt_tokens).detach().cpu().numpy()
    img = np.zeros((*correct.shape, 3), dtype=np.uint8)
    img[correct] = np.array([50, 180, 80], dtype=np.uint8)
    img[~correct] = np.array([190, 60, 60], dtype=np.uint8)
    return Image.fromarray(img)


def confidence_to_pil(confidence: torch.Tensor) -> Image.Image:
    arr = np.clip(confidence.detach().cpu().numpy(), 0.0, 1.0)
    rgb = np.stack([
        (255 * arr).astype(np.uint8),
        (255 * (1.0 - np.abs(arr - 0.5) * 2.0)).astype(np.uint8),
        (255 * (1.0 - arr)).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(rgb)


def fixed_frame_indices(rollout_steps: int, desired: Iterable[int] | None = None) -> list[int]:
    desired = list(desired or CONTACT_SHEET_COLS)
    out = []
    for idx in desired:
        if 0 <= idx < rollout_steps and idx not in out:
            out.append(idx)
    if rollout_steps - 1 not in out:
        out.append(rollout_steps - 1)
    return out


def _sample_topk_tokens(topk_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    b, k, h, w = topk_probs.shape
    flat_probs = topk_probs.permute(0, 2, 3, 1).reshape(b * h * w, k)
    flat_indices = topk_indices.permute(0, 2, 3, 1).reshape(b * h * w, k)
    if flat_probs.device.type == "mps":
        sampled = torch.multinomial(flat_probs.cpu(), 1).to(flat_probs.device)
    else:
        sampled = torch.multinomial(flat_probs, 1)
    return flat_indices[torch.arange(b * h * w, device=flat_indices.device), sampled.flatten()].reshape(b, h, w)


def make_tiled_image(
    rows: list[tuple[str, list[Image.Image]]],
    col_labels: list[str],
    *,
    title: str | None = None,
    cell_pad: int = 6,
    label_w: int = 180,
    header_h: int = 34,
) -> Image.Image:
    font = ImageFont.load_default()
    cell_w = max(img.width for _, imgs in rows for img in imgs)
    cell_h = max(img.height for _, imgs in rows for img in imgs)
    ncols = len(col_labels)
    title_h = 28 if title else 0
    width = label_w + ncols * (cell_w + cell_pad) + cell_pad
    height = title_h + header_h + len(rows) * (cell_h + cell_pad) + cell_pad
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    y = cell_pad
    if title:
        draw.text((cell_pad, y), title, fill=(240, 240, 240), font=font)
        y += title_h

    for i, label in enumerate(col_labels):
        x = label_w + cell_pad + i * (cell_w + cell_pad)
        draw.text((x + 4, y), label, fill=(220, 220, 220), font=font)
    y += header_h

    for row_label, imgs in rows:
        draw.text((cell_pad, y + 4), row_label, fill=(230, 230, 230), font=font)
        for i, img in enumerate(imgs):
            x = label_w + cell_pad + i * (cell_w + cell_pad)
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), resample=Image.NEAREST)
            canvas.paste(img, (x, y))
        y += cell_h + cell_pad
    return canvas


def make_contact_sheet(row_groups, row_labels, col_labels, col_indices, edge_rows=None):
    edge_rows = edge_rows or set()
    rows = []
    for row_idx, (label, group) in enumerate(zip(row_labels, row_groups)):
        imgs = []
        for idx in col_indices:
            t = group[idx]
            imgs.append(edge_to_pil(t) if row_idx in edge_rows else tensor_to_pil(t))
        rows.append((label, imgs))
    return make_tiled_image(rows, col_labels)


def make_strip(frames: torch.Tensor, action_label: str) -> Image.Image:
    labels = [f"f{i}" for i in range(frames.shape[0])]
    rows = [(f"action:{action_label}", [tensor_to_pil(frames[i]) for i in range(frames.shape[0])])]
    return make_tiled_image(rows, labels)


def action_name_map(device: torch.device) -> dict[str, torch.Tensor]:
    return get_test_actions(device)


@torch.no_grad()
def load_context_window(
    model,
    vqvae,
    context_path: str,
    *,
    start_idx: int,
    rollout_steps: int,
    device: torch.device,
    prime_buffer_enabled: bool = True,
):
    data = np.load(context_path)
    tokens = torch.from_numpy(data["tokens"][start_idx:start_idx + rollout_steps + 1]).long().to(device)
    actions = torch.from_numpy(data["actions"][start_idx:start_idx + rollout_steps]).float().to(device)
    gt_tokens = tokens[1:]
    gt_rgb = vqvae.decode_code(gt_tokens)
    recon_rgb = gt_rgb

    prime_len = min(getattr(model, "temporal_context_len", 8), start_idx) if prime_buffer_enabled else 0
    if prime_len > 0:
        hist_tokens = torch.from_numpy(data["tokens"][start_idx - prime_len:start_idx + 1]).long().to(device)
        hist_actions = torch.from_numpy(data["actions"][start_idx - prime_len:start_idx]).float().to(device)
        temporal_buffer = prime_temporal_buffer(model, hist_tokens, hist_actions)
    else:
        temporal_buffer = []

    return {
        "context_name": os.path.basename(context_path),
        "tokens": tokens,
        "actions": actions,
        "z_start": tokens[0:1],
        "gt_tokens": gt_tokens,
        "gt_rgb": gt_rgb,
        "recon_rgb": recon_rgb,
        "temporal_buffer": temporal_buffer,
        "prime_len": prime_len,
    }


@torch.no_grad()
def rollout_sequence(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_seq: torch.Tensor,
    *,
    topk: int,
    temporal_buffer: list[torch.Tensor] | None = None,
    oracle_tokens: torch.Tensor | None = None,
    gt_tokens: torch.Tensor | None = None,
) -> dict:
    temporal_buffer = list(temporal_buffer or [])
    z_t = z_start
    pred_tokens = []
    pred_rgb = []
    argmax_unique = []
    mean_max_prob = []
    token_accuracy = []
    confidence_maps = []

    for t in range(action_seq.shape[0]):
        logits, new_state = model.step(z_t, action_seq[t:t + 1], temporal_buffer)
        probs = F.softmax(logits, dim=1)
        greedy = logits.argmax(dim=1)
        argmax_unique.append(int(greedy.unique().numel()))
        confidence_maps.append(probs.max(dim=1)[0][0].detach().cpu())
        mean_max_prob.append(float(probs.max(dim=1)[0].mean().item()))
        if gt_tokens is not None:
            token_accuracy.append(float((greedy == gt_tokens[t:t + 1]).float().mean().item()))

        if oracle_tokens is not None:
            z_next = oracle_tokens[t:t + 1]
        elif topk > 0:
            topk_probs, topk_indices = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
            b, _, h, w = topk_probs.shape
            z_next = _sample_topk_tokens(topk_probs, topk_indices)
        else:
            z_next = greedy

        pred_tokens.append(z_next.detach().clone())
        pred_rgb.append(vqvae.decode_code(z_next))
        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > getattr(model, "temporal_context_len", 8):
            temporal_buffer.pop(0)
        z_t = z_next

    return {
        "pred_tokens": torch.stack(pred_tokens, dim=0),
        "pred_rgb": torch.cat(pred_rgb, dim=0),
        "argmax_unique_codes": argmax_unique,
        "mean_max_prob": mean_max_prob,
        "token_accuracy_per_frame": token_accuracy,
        "confidence_maps": confidence_maps,
    }


@torch.no_grad()
def run_primed_rollout(model, vqvae, context_path: str, start_idx: int, rollout_steps: int, topk: int, device: torch.device):
    ctx = load_context_window(model, vqvae, context_path, start_idx=start_idx, rollout_steps=rollout_steps, device=device, prime_buffer_enabled=True)
    run = rollout_sequence(model, vqvae, ctx["z_start"], ctx["actions"], topk=topk, temporal_buffer=ctx["temporal_buffer"], gt_tokens=ctx["gt_tokens"])
    return run["pred_tokens"], run["pred_rgb"], ctx["gt_tokens"], ctx["gt_rgb"], ctx["temporal_buffer"]


@torch.no_grad()
def run_oracle_rollout(model, vqvae, gt_tokens: torch.Tensor, action_seq: torch.Tensor, topk: int, temporal_buffer=None):
    run = rollout_sequence(model, vqvae, gt_tokens[0:1], action_seq, topk=topk, temporal_buffer=temporal_buffer, oracle_tokens=gt_tokens, gt_tokens=gt_tokens)
    return run["pred_rgb"]


def compute_decoded_target_reference(vqvae, gt_tokens: torch.Tensor, gt_rgb: torch.Tensor, lpips_fn) -> dict:
    return {
        "reference_type": "decoded_target_tokens",
        "reference_note": "Reference frame row is VQ-VAE decode(gt_tokens); scalar self-comparison metrics are omitted because they are uninformative.",
    }


def compute_quality_summary(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, lpips_fn, edge_threshold: float) -> dict:
    return compute_metrics(pred_rgb, gt_rgb, lpips_fn, edge_threshold=edge_threshold)


def edge_flicker_proxy(pred_rgb: torch.Tensor) -> list[float]:
    edges = _edge_map(pred_rgb)
    if edges.shape[0] < 2:
        return []
    delta = (edges[1:] - edges[:-1]).abs().mean(dim=(1, 2, 3))
    return [float(x) for x in delta.cpu().tolist()]
