import torch as th

from envs.dflex_env import DFlexEnv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

import matplotlib.pyplot as plt

DIM = 8

class TestEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=1024, seed=0, episode_length=10, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False, dim=DIM):

        num_obs = dim + dim
        num_act = dim

        super(TestEnv, self).__init__(num_envs, num_obs, num_act, episode_length, 1, seed, no_grad, render, device)

        self.dim = dim
        self.stochastic_init = False
        self.early_termination = False

        self.A = th.rand((dim, dim), dtype=th.float32, device=device) * 0.3 # * 1.1
        self.B = th.rand((dim, dim), dtype=th.float32, device=device) * 0.3 # * 1.1
        
        # self.B = th.eye(dim, dtype=th.float32, device=device)
        self.position = th.rand((num_envs, dim), dtype=th.float32, device=device) * 10.
        self.destination = th.rand((num_envs, dim), dtype=th.float32, device=device) * 10.

    def step(self, actions: th.Tensor):
        with df.ScopedTimer("simulate", active=False, detailed=False):
            actions = actions.view((self.num_envs, self.num_actions))
            self.actions = actions
            
            # forward;
            self.position = th.matmul(self.A.unsqueeze(0), self.position.unsqueeze(-1)).squeeze(-1) + \
                            th.matmul(self.B.unsqueeze(0), self.actions.unsqueeze(-1)).squeeze(-1)
            
        self.reset_buf = th.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        #self.obs_buf_before_reset = self.obs_buf.clone()

        with df.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)
                
        # with df.ScopedTimer("render", active=False, detailed=False):
        #     self.render()

        #self.extras = {'obs_before_reset': self.obs_buf_before_reset}
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self, mode = 'human'):
        
        return
    
    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = th.arange(self.num_envs, dtype=th.long, device=self.device)
                
        if env_ids is not None:

            self.position[env_ids, :] = th.rand((len(env_ids), self.dim), dtype=th.float32, device=self.device) * 10.
            self.destination[env_ids, :] = th.rand((len(env_ids), self.dim), dtype=th.float32, device=self.device) * 10.

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf

    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self):
        
        self.position = self.position.detach()

    '''
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        
        obs_buf = th.zeros_like(self.obs_buf)
        obs_buf[:, :self.dim] = self.position
        obs_buf[:, self.dim:] = self.destination
        
        self.obs_buf = obs_buf

    def calculateReward(self):

        self.rew_buf = th.norm(self.destination - self.position, p=2, dim=-1)
        self.rew_buf = -th.square(self.rew_buf)

        # reset agents
        self.reset_buf = th.where(self.progress_buf > self.episode_length - 1, th.ones_like(self.reset_buf), self.reset_buf)