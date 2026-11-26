"""Resample recorded joint demonstrations without changing the source Zarr."""

import math

import numpy as np
import zarr

from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer


class IndexedArray:
    """Read a lower-rate phase view from the original Zarr without copying images."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = indices
        self.shape = (len(indices),) + source.shape[1:]
        self.dtype = source.dtype

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, *rest = item
            mapped = self.indices[rows]
            if np.isscalar(mapped):
                return self.source[int(mapped)][tuple(rest)]
            if len(mapped) == 0:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)[(slice(None), *rest)]
            result = self.source.oindex[mapped]
            return result[(slice(None), *rest)]
        rows = self.indices[item]
        if np.isscalar(rows):
            return self.source[int(rows)]
        if len(rows) == 0:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        return self.source.oindex[rows]


def load_joint_replay(zarr_path, keys, action_key, sample_hz, sample_phase=None):
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
    if sample_phase is not None and (
        isinstance(sample_phase, bool)
        or not isinstance(sample_phase, int)
        or not 0 <= sample_phase < stride
    ):
        raise ValueError(f"sample_phase must be an integer in [0, {stride})")
    # Each source episode supplies one stream per downsampling phase. Keep phases
    # as separate episodes so no training window crosses a phase boundary.
    selected, lengths, groups = [], [], []
    start = 0
    for group, end in enumerate(source["meta/episode_ends"][:]):
        phases = range(stride) if sample_phase is None else (sample_phase,)
        for phase in phases:
            indices = np.arange(start + phase, int(end) - stride, stride, dtype=np.int64)
            if len(indices):
                selected.append(indices)
                lengths.append(len(indices))
                groups.append(group)
        start = int(end)
    if not selected:
        raise ValueError(
            "no episode contains a valid future command at the training rate"
        )
    indices = np.concatenate(selected)
    data = {key: IndexedArray(source["data"][key], indices)
            for key in keys if key != action_key}
    # Raw command[t] is concurrent with observation[t]. The training action
    # label is the command one training step later, never from another episode.
    commands = IndexedArray(source["data"][action_key], indices + stride)
    # Relative action is window-dependent: the entire chunk uses the slow
    # observation at its start as one fixed anchor, so it is applied in the
    # dataset after SequenceSampler selects a window.
    data[action_key] = commands
    replay = ReplayBuffer(
        {
            "data": data,
            "meta": {"episode_ends": np.cumsum(lengths, dtype=np.int64)},
        }
    )
    replay.source_episode_ids = np.asarray(groups, dtype=np.int64)
    return replay
