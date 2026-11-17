from typing import Dict
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import copy
import tqdm
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from reactive_diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)
from reactive_diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=True,
                 transform_params=None,
                 sample_hz=None,
                 slow_obs_latency_steps=0,
                 ):
        assert os.path.isdir(dataset_path)
        self.action_key = shape_meta['action'].get('key', 'action')
        self.slow_obs_latency_steps = int(slow_obs_latency_steps)
        prefix_length = n_obs_steps if n_obs_steps is not None else horizon + n_latency_steps
        if self.slow_obs_latency_steps != slow_obs_latency_steps or not 0 <= self.slow_obs_latency_steps < prefix_length:
            raise ValueError('slow_obs_latency_steps must be an integer within the observation prefix')
        self.action_space = shape_meta['action'].get('space', 'cartesian')
        if self.action_space not in ('cartesian', 'joint', 'tcp_delta'):
            raise ValueError(f"unsupported action space: {self.action_space}")
        if self.action_space in ('joint', 'tcp_delta') and (delta_action or relative_action):
            raise ValueError("ForceGello joint and TCP delta targets require delta_action=False and relative_action=False")

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)

        zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + [self.action_key])
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        if sample_hz is not None:
            if self.action_space not in ('joint', 'tcp_delta') or delta_action or relative_action:
                raise ValueError("sample_hz resampling requires ForceGello joint or TCP delta actions")
            from reactive_diffusion_policy.dataset.forcegello_sampling import load_joint_replay
            replay_buffer = load_joint_replay(zarr_path, zarr_load_keys, self.action_key, float(sample_hz))
        else:
            replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=zarr_load_keys)

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer[self.action_key][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer[self.action_key][:] = actions_diff
        
        self.relative_action = relative_action
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action
        self.transforms = RealWorldTransforms(option=transform_params)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps
        self.key_first_k = key_first_k

        self.seed = seed
        source_ids = getattr(replay_buffer, 'source_episode_ids', None)
        if source_ids is None:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
            train_mask = downsample_mask(
                mask=~val_mask, max_n=max_train_episodes, seed=seed)
        else:
            # All 20 Hz phases of one recording must stay in the same split.
            _, group_indices = np.unique(source_ids, return_inverse=True)
            group_val = get_val_mask(
                n_episodes=int(group_indices.max()) + 1,
                val_ratio=val_ratio, seed=seed)
            group_train = downsample_mask(
                mask=~group_val, max_n=max_train_episodes, seed=seed)
            val_mask = group_val[group_indices]
            train_mask = group_train[group_indices]

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys:
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        if self.relative_action:
            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
                for key in relative_data_dict.keys():
                    if key == 'action':
                        relative_data_dict[key].append(data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
            relative_data_dict = dict_apply(relative_data_dict, np.stack)

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
        else:
            action_all = self.replay_buffer[self.action_key][:, :self.shape_meta['action']['shape'][0]]

        if self.action_space in ('joint', 'tcp_delta'):
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(action_all)
        else:
            normalizer['action'] = get_action_normalizer(action_all)

        # obs
        for key in list(set(self.lowdim_keys)):
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])

        for key in list(set(self.extended_lowdim_keys)):
            if key in self.lowdim_keys:
                assert self.shape_meta['extended_obs'][key]['shape'][0] == self.shape_meta['obs'][key]['shape'][0], \
                    f"Extended obs {key} has different shape from obs {key}"
            else:
                if self.relative_action and key in relative_data_dict:
                    normalizer[key] = get_action_normalizer(relative_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' in key:
                    normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                else:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])

        # image
        for key in list(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def _observation_indices(self, key: str, sample_index: int, prefix_steps: int) -> np.ndarray:
        """Per-modality history for the slow model, before its latent becomes usable.

        Optional hold_steps simulates the slow camera's sample-and-hold on the
        episode clock, including image age between visual updates.
        """
        meta = self.shape_meta['obs'][key]
        stride = int(meta.get('temporal_downsample_ratio', self.obs_downsample_ratio))
        count = meta.get('n_obs_steps')
        if stride < 1 or (count is not None and int(count) < 1):
            raise ValueError(f"invalid observation sampling for {key}")
        if count is None:
            indices = np.arange(prefix_steps - 1, -1, -stride)[::-1]
        else:
            indices = prefix_steps - 1 - np.arange(int(count) - 1, -1, -1) * stride
        indices = indices - self.slow_obs_latency_steps
        if np.any(indices < 0):
            raise ValueError(f"observation history too short for {key}")
        hold = int(meta.get('hold_steps', 1))
        if hold < 1:
            raise ValueError(f"invalid camera hold interval for {key}")
        if hold > 1:
            buffer_start, buffer_end, sample_start, _ = self.sampler.indices[sample_index]
            ends = self.replay_buffer.episode_ends[:]
            episode_index = np.searchsorted(ends, buffer_start, side='right')
            episode_start = 0 if episode_index == 0 else ends[episode_index - 1]
            absolute = np.clip(buffer_start + indices - sample_start, buffer_start, buffer_end - 1)
            held = episode_start + ((absolute - episode_start) // hold) * hold
            indices = held - buffer_start + sample_start
            if np.any(indices >= prefix_steps):
                raise ValueError(f"camera hold for {key} falls after the observed prefix")
        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        prefix_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon + self.n_latency_steps

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            indices = self._observation_indices(key, idx, prefix_steps)
            if np.any(indices < 0):
                # A held 5 Hz frame can precede the 8-step action window after
                # delaying the slow observation by one 20 Hz cycle.
                buffer_start, _, sample_start, _ = self.sampler.indices[idx]
                absolute = buffer_start + indices - sample_start
                frames = np.stack([self.replay_buffer[key][int(i)] for i in absolute])
            else:
                frames = data[key][indices]
            obs_dict[key] = np.moveaxis(frames,-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                indices = self._observation_indices(key, idx, prefix_steps)
                obs_dict[key] = data[key][indices, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
                # save ram
                if key not in self.extended_lowdim_keys:
                    del data[key]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in ['left_robot_wrt_right_robot_tcp_pose', 'right_robot_wrt_left_robot_tcp_pose']:
            if key in obs_dict:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                del data[key]

        action = data[self.action_key][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations already use the selected history prefix
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=obs_dict[key][-1])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy)
        }
        return torch_data

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_emb_absolute_12fps'])
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()

    for i in range(len(dataset)):
        data = dataset[i]

if __name__ == '__main__':
    test()
