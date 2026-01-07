from __future__ import annotations

import numpy as np


def _yaw_bin_index(yaw_raw: float) -> int:
    yaw_raw = float(np.clip(yaw_raw, -1.0, 1.0))
    if yaw_raw <= -0.5:
        return 0  # Hard Left
    if yaw_raw <= -0.1:
        return 1  # Left
    if yaw_raw <= 0.1:
        return 2  # Center
    if yaw_raw <= 0.5:
        return 3  # Right
    return 4  # Hard Right


def _pitch_bin_index(pitch_raw: float) -> int:
    pitch_raw = float(np.clip(pitch_raw, -1.0, 1.0))
    if pitch_raw <= -0.2:
        return 0  # Down
    if pitch_raw <= 0.2:
        return 1  # Level
    return 2  # Up


def encode_action_v5_np(
    *,
    yaw_raw: float = 0.0,
    pitch_raw: float = 0.0,
    w: float | bool = 0.0,
    a: float | bool = 0.0,
    s: float | bool = 0.0,
    d: float | bool = 0.0,
    jump: float | bool = 0.0,
    sprint: float | bool = 0.0,
    sneak: float | bool = 0.0,
) -> np.ndarray:
    """
    v5.0: Build a (15,) float32 discrete action vector:
      [yaw(5 one-hot), pitch(3 one-hot), W, A, S, D, jump, sprint, sneak]
    """
    action = np.zeros((15,), dtype=np.float32)

    action[_yaw_bin_index(yaw_raw)] = 1.0
    action[5 + _pitch_bin_index(pitch_raw)] = 1.0

    action[8] = 1.0 if bool(w) else 0.0
    action[9] = 1.0 if bool(a) else 0.0
    action[10] = 1.0 if bool(s) else 0.0
    action[11] = 1.0 if bool(d) else 0.0
    action[12] = 1.0 if bool(jump) else 0.0
    action[13] = 1.0 if bool(sprint) else 0.0
    action[14] = 1.0 if bool(sneak) else 0.0

    return action

