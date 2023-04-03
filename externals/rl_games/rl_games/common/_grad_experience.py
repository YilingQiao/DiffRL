import torch

from rl_games.common.experience import ExperienceBuffer

class GradExperienceBuffer(ExperienceBuffer):

    def _init_from_env_info(self, env_info):

        super()._init_from_env_info(env_info)

        # store first and second order analytical gradients of advantage w.r.t. actions;
        
        base_shape = self.obs_base_shape
        action_shape = self.actions_shape
        dtype = torch.float32
        device = self.device
        
        self.tensor_dict['adv_gradient'] = torch.zeros(base_shape + action_shape, dtype=dtype, device=device)
        self.tensor_dict['adv_hessian'] = torch.zeros(base_shape + action_shape + action_shape, dtype=dtype, device=device)