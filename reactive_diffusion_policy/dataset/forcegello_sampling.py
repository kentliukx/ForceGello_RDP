"""Resample recorded joint demonstrations without changing the source Zarr."""

import math

import numpy as np
import zarr

from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer


def load_joint_replay(zarr_path, keys, action_key, sample_hz):
    source = zarr.open_group(zarr_path, mode="r")
    source_hz = float(source.attrs["sample_hz"])
    if not all(math.isfinite(rate) and rate > 0 for rate in (source_hz, sample_hz)):
        raise ValueError("source and training sample rates must be finite and positive")
    ratio = source_hz / sample_hz
    if ratio < 1 or not math.isclose(ratio, round(ratio)):
        raise ValueError(
            "source sample_hz must be an integer multiple of training sample_hz"
        )
    stride = round(ratio)
    if stride == 1:
        return ReplayBuffer.copy_from_path(zarr_path, keys=keys)

    target_key = (
        "target"
        if action_key == "action"
        else action_key.removesuffix("action") + "target"
    )
    if target_key not in source["data"]:
        raise ValueError(
            f"resampling joint actions requires current leader targets: {target_key}"
        )
    # Reserve a real future selected frame in each episode; never label across a gap.
    selected, lengths = [], []
    start = 0
    for end in source["meta/episode_ends"][:]:
        indices = np.arange(start, int(end) - stride, stride, dtype=np.int64)
        if len(indices):
            selected.append(indices)
            lengths.append(len(indices))
        start = int(end)
    if not selected:
        raise ValueError(
            "no episode contains a valid next-frame target at the training rate"
        )
    indices = np.concatenate(selected)
    data = {
        key: source["data"][key].oindex[indices] for key in keys if key != action_key
    }
    # Striding old action[t] would still predict the next 60 Hz frame (16.7 ms).
    # Rebuild from leader targets at t + stride, i.e. 50 ms for 60 -> 20 Hz.
    data[action_key] = source["data"][target_key].oindex[indices + stride]
    return ReplayBuffer(
        {
            "data": data,
            "meta": {"episode_ends": np.cumsum(lengths, dtype=np.int64)},
        }
    )
