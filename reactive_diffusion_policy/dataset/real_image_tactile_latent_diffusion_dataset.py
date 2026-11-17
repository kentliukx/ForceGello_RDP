import einops
import numpy as np
import tqdm
import copy
import torch
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class RealImageTactileLatentDiffusionDataset(RealImageTactileDataset):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []
        if self.action_space in ('joint', 'tcp_delta') and not self.relative_action:
            # The latent statistics need only actions. Reading two camera images
            # for every overlapping window would turn the lazy three-phase view
            # into a very expensive full-dataset image scan.
            action_values = self.replay_buffer[self.action_key][
                :, :self.shape_meta['action']['shape'][0]
            ]
            action_buffer = ReplayBuffer({
                'data': {self.action_key: action_values},
                'meta': {'episode_ends': self.replay_buffer.episode_ends[:]},
            })
            action_sampler = copy.copy(self.sampler)
            action_sampler.replay_buffer = action_buffer
            action_sampler.keys = [self.action_key]
            def iter_actions():
                for idx in range(len(action_sampler)):
                    values = action_sampler.sample_sequence(idx)[self.action_key]
                    if self.n_latency_steps:
                        values = values[self.n_latency_steps:]
                    yield torch.from_numpy(values.astype(np.float32))
            actions = iter_actions()
        else:
            actions = (sample['action'] for sample in self)

        for action in tqdm.tqdm(actions, total=len(self), leave=False,
                                desc='Calculating latent action for normalizer'):
            action = action.to(self.at.device).unsqueeze(0)
            action = normalizer['action'].normalize(action)
            latent_action = self.at.encoder(
                self.at.preprocess(action / self.at.act_scale)
            )
            if self.at.use_vq:
                if not self.use_latent_action_before_vq:
                    latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
            else:
                latent_action, _ = self.at.quant_state_without_vq(latent_action)
            if self.at.use_conv_encoder:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
            else:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
            latent_action_all.append(latent_action[0].cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer
