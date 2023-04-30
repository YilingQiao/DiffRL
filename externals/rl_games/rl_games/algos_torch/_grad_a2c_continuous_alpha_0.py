'''
This script was used to test original alpha policy,
which has some theoretical flaws. (Hessian problem)
'''

import time

import torch 
from torch import nn
import numpy as np
import gym

from rl_games.algos_torch import running_mean_std, torch_ext

from rl_games.common.a2c_common import swap_and_flatten01, A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common._grad_experience import GradExperienceBuffer
from rl_games.common import common_losses
from rl_games.algos_torch._grad_distribution import GradNormal

import copy
from utils.running_mean_std import RunningMeanStd

class GradA2CAgent(A2CAgent):
    def __init__(self, base_name, config):
        
        super().__init__(base_name, config)

        # unsupported settings;

        if self.use_action_masks or \
            self.has_central_value or \
            self.has_self_play_config or \
            self.self_play or \
            self.rnn_states or \
            self.has_phasic_policy_gradients or \
            isinstance(self.observation_space, gym.spaces.Dict):

            raise NotImplementedError()

        # we have additional hyperparameter [gi_step_num] that determines differentiable step size;
        # this step size is also used for policy update based on analytical policy gradients;
        self.gi_num_step = config['gi_params']['num_step']

        # change to proper running mean std for backpropagation;
        if self.normalize_input:
            if isinstance(self.observation_space, gym.spaces.Dict):
                raise NotImplementedError()
            else:
                self.obs_rms = RunningMeanStd(shape=self.obs_shape, device=self.ppo_device)
                
        if self.normalize_value:
            raise NotImplementedError()

        # episode length;
        self.episode_max_length = self.vec_env.env.episode_length
        self.max_alpha = 1e-1

    def init_tensors(self):
        
        super().init_tensors()

        # use specialized experience buffer;
        
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }

        self.experience_buffer = GradExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        # add advantage gradient and hessian;
        self.tensor_list = self.tensor_list + ['adv_gradient', 'adv_hessian']
        
        
    def train_epoch(self):

        A2CBase.train_epoch(self)

        play_time_start = time.time()

        # collect experience;
        if self.is_rnn:
            raise NotImplementedError()
        else:
            batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            raise NotImplementedError()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        if self.is_rnn:
            raise NotImplementedError()
        
        if True:
            preupdate_action_eps_jac = self.action_eps_jacobian(self.dataset.values_dict['mu'],
                                                                self.dataset.values_dict['sigma'],
                                                                self.dataset.values_dict['rp_eps'])
            preupdate_action_eps_jacdet = torch.linalg.det(preupdate_action_eps_jac)

        for _ in range(0, self.mini_epochs_num):

            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)   

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == 'standard':
                raise NotImplementedError()
            kls.append(av_kls)
        
        if True:    
            tmp_actions_batch = self.dataset.values_dict['actions']
            tmp_obs_batch = self.dataset.values_dict['obs']
            
            tmp_batch_dict = {
                'is_train': True,
                'prev_actions': tmp_actions_batch, 
                'obs' : tmp_obs_batch,
            }
            tmp_res_dict = self.model(tmp_batch_dict)
            
            post_mu = tmp_res_dict['mus']
            post_sigma = tmp_res_dict['sigmas']
            postupdate_action_eps_jac = self.action_eps_jacobian(post_mu,
                                                                 post_sigma,
                                                                self.dataset.values_dict['rp_eps'])
            postupdate_action_eps_jacdet = torch.linalg.det(postupdate_action_eps_jac)

            est_alpha_det = postupdate_action_eps_jacdet / preupdate_action_eps_jacdet
            est_alpha_ratio = est_alpha_det / self.dataset.values_dict['alpha_policy_det']
            self.writer.add_scalar("info/est_alpha_ratio", est_alpha_ratio.mean(), self.frame)

        if self.schedule_type == 'standard_epoch':
            raise NotImplementedError()

        if self.has_phasic_policy_gradients:
            raise NotImplementedError()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def action_eps_jacobian(self, mu, sigma, eps):
        
        distr = GradNormal(mu, sigma)
        eps.requires_grad = True
        actions = distr.eps_to_action(eps)
        
        jacobian = torch.zeros((eps.shape[0], actions.shape[1], eps.shape[1]))
        
        for d in range(actions.shape[1]):
            target = torch.sum(actions[:, d])
            grad = torch.autograd.grad(target, eps, retain_graph=True)
            grad = torch.stack(grad)
            jacobian[:, d, :] = grad
            
        return jacobian

    def neglogp(self, x, mean, std, logstd):

        assert x.ndim == 2 and mean.ndim == 2 and std.ndim == 2 and logstd.ndim == 2, ""
        # assert x.shape[0] == mean.shape[0] and x.shape[0] == std.shape[0] and x.shape[0] == logstd.shape[0], ""

        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)
            
    def forward_model(self, input_dict):
            
        is_train = input_dict.get('is_train', True)
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, states = self.model.a2c_network(input_dict)
        sigma = torch.exp(logstd)
        distr = GradNormal(mu, sigma) # torch.distributions.Normal(mu, sigma)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = self.model.neglogp(prev_actions, mu, sigma, logstd)
            result = {
                'prev_neglogp' : torch.squeeze(prev_neglogp),
                'values' : value,
                'entropy' : entropy,
                'rnn_states' : states,
                'mus' : mu,
                'sigmas' : sigma
            }                
            return result
        else:
            selected_eps = distr.sample_eps()
            selected_action = distr.eps_to_action(selected_eps)
            neglogp = self.model.neglogp(selected_action, mu, sigma, logstd)
            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : value,
                'actions' : selected_action,
                'rnn_states' : states,
                'mus' : mu,
                'sigmas' : sigma,
                'rp_distr' : distr,
                'rp_eps' : selected_eps,
            }
            return result

    def get_action_values(self, obs, obs_rms):
        
        # normalize input if needed, we update rms only here;
        processed_obs = obs['obs']
        if self.normalize_input:
            # update rms;
            with torch.no_grad():
                self.obs_rms.update(processed_obs)
            processed_obs = obs_rms.normalize(processed_obs)
        
        # [std] is a vector of length [action_dim], which is shared by all the envs;
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
        
        res_dict = self.forward_model(input_dict)
        res_dict['obs'] = processed_obs
        
        # assert res_dict['rnn_states'] == None, "Not supported yet"
        assert not self.has_central_value, "Not supported yet"

        if self.normalize_value:
            raise NotImplementedError()

        return res_dict

    def get_critic_values(self, obs, use_target_critic: bool, obs_rms_train: bool):

        if use_target_critic:   
            critic = self.target_critic
            # critic.eval()
        else:
            critic = self.critic

        if self.normalize_input:

            if obs_rms_train:
                self.running_mean_std.train()
            else:
                self.running_mean_std.eval()

        processed_obs = self._preproc_obs(obs)
        values = critic(processed_obs)

        if self.normalize_value:
            values = self.value_mean_std(values, True)

        return values

    def play_steps(self):

        '''
        Unlike PPO, here we conduct several actor & critic network updates using gradient descent. 
        '''

        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        # indicator for steps that grad computation starts;
        grad_start = torch.zeros_like(self.experience_buffer.tensor_dict['dones'])
        
        grad_obses = []
        grad_values = []
        grad_next_values = []
        grad_actions = []
        grad_rewards = []
        grad_fdones = []
        grad_rp_eps = []

        # use frozen [obs_rms] during this one function call;
        curr_obs_rms = None
        if self.normalize_input:
            with torch.no_grad():
                curr_obs_rms = copy.deepcopy(self.obs_rms)

        # start with clean grads;
        self.obs = self.vec_env.env.initialize_trajectory()
        self.obs = self.obs_to_tensors(self.obs)
        grad_start[0, :] = 1.0

        for n in range(self.gi_num_step):

            if n > 0:
                grad_start[n, :] = self.dones

            # get action for current observation;
            if self.use_action_masks:
                raise NotImplementedError()
            else:
                res_dict = self.get_action_values(self.obs, curr_obs_rms)

            # we store tensor objects with gradients;
            grad_obses.append(res_dict['obs'])
            grad_values.append(res_dict['values'])
            grad_actions.append(res_dict['actions'])
            grad_fdones.append(self.dones.float())
            grad_rp_eps.append(res_dict['rp_eps'])

            # [obs] is an observation of the current time step;
            # store processed obs, which might have been normalized already;
            self.experience_buffer.update_data('obses', n, res_dict['obs'])

            # [dones] indicate if this step is the start of a new episode;
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                raise NotImplementedError()

            # take action, which goes through [tanh] to 
            # make its value located in between [0, 1];
            step_time_start = time.time()
            
            # grad_actions[-1] = grad_actions[-1].detach()
            # grad_actions[-1].requires_grad = True
            grad_actions[-1].retain_grad()
            # actions = torch.tanh(grad_actions[-1])
            actions = grad_actions[-1]
            self.obs, rewards, self.dones, infos = self.vec_env.step(actions)
            
            self.obs = self.obs_to_tensors(self.obs)
            rewards = rewards.unsqueeze(-1)
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            # compute value of next state;
            # @TODO: It slows down the code...
            if True:
                # assume that 'obs_before_reset' == 'obs' if the episode is not done yet;
                # sanity check for above condition;
                '''
                for i in range(len(self.obs['obs'])):
                    o = self.obs['obs'][i]
                    no = infos['obs_before_reset'][i]
                    diff = torch.norm(o - no)
                    if diff > 1e-5:
                        assert self.dones[i], ""
                '''
                
                next_obs = infos['obs_before_reset']
                if self.normalize_input:
                    # do not update rms here;
                    next_obs = curr_obs_rms.normalize(next_obs)
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : next_obs,
                    'rnn_states' : self.rnn_states
                }
                result = self.model(input_dict)
                next_value: torch.Tensor = result['values']
                if self.normalize_value:
                    raise NotImplementedError()
                # next_value = next_value.zero_()
                grad_next_values.append(next_value)

            done_env_ids = self.dones.nonzero(as_tuple = False).squeeze(-1)
            for id in done_env_ids:
                if torch.isnan(infos['obs_before_reset'][id]).sum() > 0 \
                    or torch.isinf(infos['obs_before_reset'][id]).sum() > 0 \
                    or (torch.abs(infos['obs_before_reset'][id]) > 1e6).sum() > 0: # ugly fix for nan values
                    grad_next_values[-1][id] = 0.
                elif self.current_lengths[id] < self.episode_max_length - 1: # early termination
                    grad_next_values[-1][id] = 0.
            
            # add default reward; @TODO: add reward shaper?
            grad_rewards.append(rewards)

            # do not use reward shaper for now;
            if self.value_bootstrap and 'time_outs' in infos:
                raise NotImplementedError()
                
            self.experience_buffer.update_data('rewards', n, rewards)

            self.current_rewards += rewards.detach()
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        '''
        Compute gradients of advantage terms and use them to define alpha-policy.
        '''

        last_fdones = self.dones.float()
        
        # compute advantages using GAE;
        
        grad_advs = self.grad_advantages(self.tau, 
                                        grad_values, 
                                        grad_next_values,
                                        grad_rewards,
                                        grad_fdones,
                                        last_fdones)

        if True:
    
            # compute gradients of advantages;
            
            adv_gradient, adv_hessian = self.differentiate_grad_advantages(grad_actions,
                                                                           grad_advs,
                                                                           grad_start,
                                                                           True)
            
            self.experience_buffer.tensor_dict['adv_gradient'] = adv_gradient
            self.experience_buffer.tensor_dict['adv_hessian'] = adv_hessian

        self.clear_experience_buffer_grads()

        with torch.no_grad():

            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

            for i in range(len(grad_advs)):
                grad_advs[i] = grad_advs[i].unsqueeze(0)
            batch_dict['advantages'] = swap_and_flatten01(torch.cat(grad_advs, dim=0).detach())
            batch_dict['rp_eps'] = swap_and_flatten01(torch.stack(grad_rp_eps, dim=0).detach())

            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time

        return batch_dict

    def grad_advantages(self, gae_tau, mb_extrinsic_values, mb_next_extrinsic_values, mb_rewards, mb_fdones, last_fdones):

        num_step = len(mb_extrinsic_values)
        mb_advs = []
        
        # GAE;

        lastgaelam = 0

        for t in reversed(range(num_step)):
            if t == num_step - 1:
                nextnonterminal = 1.0 - last_fdones
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            nextvalues = mb_next_extrinsic_values[t]

            delta = mb_rewards[t] + self.gamma * nextvalues - mb_extrinsic_values[t]
            mb_adv = lastgaelam = delta + self.gamma * gae_tau * nextnonterminal * lastgaelam
            mb_advs.append(mb_adv)

        mb_advs.reverse()
        return mb_advs

    def grad_advantages_first_terms_sum(self, grad_advs, grad_start):

        num_timestep = grad_start.shape[0]
        num_actor = grad_start.shape[1]

        adv_sum = 0

        for i in range(num_timestep):
            for j in range(num_actor):
                if grad_start[i, j]:
                    adv_sum = adv_sum + grad_advs[i][j]

        return adv_sum
    
    def differentiate_grad_advantages(self, 
                                    grad_actions: torch.Tensor, 
                                    grad_advs: torch.Tensor, 
                                    grad_start: torch.Tensor,
                                    debug: bool=False):

        '''
        Compute first-order gradients (and second-order hessians if needed) 
        of [grad_advs] w.r.t. [grad_actions] using automatic differentiation.
        '''

        num_timestep = grad_start.shape[0]
        num_actor = grad_start.shape[1]

        adv_sum: torch.Tensor = self.grad_advantages_first_terms_sum(grad_advs, grad_start)

        # compute gradients;
        
        # first-order gradient;
        
        adv_gradient = torch.autograd.grad(adv_sum, grad_actions, create_graph=True, retain_graph=True)
        adv_gradient = torch.stack(adv_gradient)
        
        # adv_sum.backward(retain_graph=True, create_graph=True)
        # adv_gradient = []
        # for ga in grad_actions:
        #     adv_gradient.append(ga.grad)
        # adv_gradient = torch.stack(adv_gradient)
        
        # unidentifiable indexing error for below code...
        
        '''
        adv_gradient = torch.autograd.grad(adv_sum, grad_actions, create_graph=True, retain_graph=True)
        adv_gradient = torch.stack(adv_gradient)
        '''
        
        adv_hessian: torch.Tensor = self.experience_buffer.tensor_dict['adv_hessian'].clone()
            
        # second-order gradient;
        
        for i in range(num_timestep):
            
            g = adv_gradient[i]         # shape = [# actor, # action dim]
            
            g_sum = g.sum(dim=0)        # shape = [# action dim]
            
            for j in range(len(g_sum)):
                
                adv_hessian[i, :, j, :] = torch.autograd.grad(g_sum[j], grad_actions[i], retain_graph=True)[0]
                
        # reweight grads;

        with torch.no_grad():

            c = (1.0 / (self.gamma * self.tau))
            cv = torch.ones((num_actor, 1), device=adv_gradient.device)

            for nt in range(num_timestep):

                # if new episode has been started, set [cv] to 1; 
                for na in range(num_actor):
                    if grad_start[nt, na]:
                        cv[na, 0] = 1.0

                adv_gradient[nt] = adv_gradient[nt] * cv
                adv_hessian[nt] = adv_hessian[nt] * cv.unsqueeze(-1)
                cv = cv * c
                
        if True:
            
            # compute gradients in brute force and compare;
            # this is to prove correctness of efficient computation of GAE-based advantage w.r.t. actions;
        
            for i in range(num_timestep):
                
                debug_adv_sum = grad_advs[i].sum()
                
                debug_grad_adv_gradient = torch.autograd.grad(debug_adv_sum, grad_actions[i], retain_graph=True, create_graph=True)[0]
                debug_grad_adv_gradient_norm = torch.norm(debug_grad_adv_gradient, p=2, dim=-1)
                
                debug_grad_error = torch.norm(debug_grad_adv_gradient - adv_gradient[i], p=2, dim=-1)
                debug_grad_error_ratio = debug_grad_error / debug_grad_adv_gradient_norm
                
                assert torch.all(debug_grad_error_ratio < 0.01), \
                    "Gradient of advantage possibly wrong"
                    
                debug_g_sum = debug_grad_adv_gradient.sum(dim=0)
                
                for j in range(len(debug_g_sum)):
                    
                    debug_grad_adv_hessian = torch.autograd.grad(debug_g_sum[j], grad_actions[i], retain_graph=True)[0]
                    debug_grad_adv_hessian_norm = torch.norm(debug_grad_adv_hessian, p=2, dim=-1)
                    
                    debug_grad_error = torch.norm(debug_grad_adv_hessian - adv_hessian[i, :, j, :], p=2, dim=-1)
                    debug_grad_error_ratio = debug_grad_error / debug_grad_adv_hessian_norm
                    
                    assert torch.all(debug_grad_error_ratio < 0.01), \
                        "Hessian of advantage possibly wrong"
                        
        adv_gradient = adv_gradient.detach()
        adv_hessian = adv_hessian.detach()

        return adv_gradient, adv_hessian

    def clear_experience_buffer_grads(self):

        '''
        Clear computation graph attached to the tensors in the experience buffer.
        '''

        with torch.no_grad():

            for k in self.experience_buffer.tensor_dict.keys():

                if not isinstance(self.experience_buffer.tensor_dict[k], torch.Tensor):

                    continue

                self.experience_buffer.tensor_dict[k] = self.experience_buffer.tensor_dict[k].detach()

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        advantages = batch_dict['advantages']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rp_eps = batch_dict['rp_eps']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        returns = advantages + values

        if self.normalize_value:
            raise NotImplementedError()
            
        with torch.no_grad():
            adv_gradient = batch_dict['adv_gradient']
            adv_hessian = batch_dict['adv_hessian']

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                raise NotImplementedError()
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        # define alpha-policy using gradient and hessian;
        # original determinant can be computed by (sign * exp(logabsdet));
        N = adv_gradient.shape[1]
        
        adv_hessian_eigvals = torch.linalg.eigvalsh(adv_hessian)
        adv_hessian_min_eigval = torch.min(adv_hessian_eigvals)
        
        if adv_hessian_min_eigval < 0:
            alpha = torch.abs(1.0 / adv_hessian_min_eigval) * 0.99
        else:
            alpha = self.max_alpha
        alpha = torch.clamp(alpha, max=self.max_alpha)
        
        # alpha = 1e-1
        print("alpha: {:.4f}".format(alpha))
        
        self.writer.add_scalar("info/alpha", alpha, self.frame)
        
        tmps, tmp = torch.linalg.slogdet(adv_hessian * alpha + torch.eye(N).unsqueeze(0))
        alpha_det = tmps * torch.exp(tmp)
        # assert torch.all(tmps > 0.), ""
        
        alpha_policy_actions = actions + alpha * adv_gradient
        alpha_policy_neglogp_actions = neglogpacs + tmp
        
        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['alpha_policy_actions'] = alpha_policy_actions
        dataset_dict['alpha_policy_neglogpacs'] = alpha_policy_neglogp_actions
        dataset_dict['alpha_policy_det'] = alpha_det
        dataset_dict['rp_eps'] = rp_eps

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            raise NotImplementedError()

    def get_full_state_weights(self):
        
        state = super().get_full_state_weights()

        if self.normalize_input:
            state['gi_obs_rms'] = self.obs_rms        
        return state

    def set_full_state_weights(self, weights):
        
        super().set_full_state_weights(weights)

        if self.normalize_input:
            self.obs_rms = weights['gi_obs_rms'].to(self.ppo_device)
    
    def calc_gradients(self, input_dict):
        
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        rpeps_batch = input_dict['rp_eps']
        obs_batch = input_dict['obs']
        
        # if self.normalize_input:
        #     obs_batch = self.obs_rms.normalize(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            # self.model.a2c_network.sigma.retain_grad()
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # LR loss;
            a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # RP loss;
            
            # RP loss computation based on second order action gradients;
            
            alpha_actions_batch = input_dict['alpha_policy_actions']
            target_alpha_action_log_probs_batch = input_dict['alpha_policy_neglogpacs']
        
            alpha_action_log_probs = self.neglogp(alpha_actions_batch, mu, sigma, torch.log(sigma))
            rp_a_loss = torch.square(target_alpha_action_log_probs_batch - alpha_action_log_probs)
            rp_a_loss = rp_a_loss.mean() * 0.5
            
            # RP loss computation based on first order action gradients;
            
            # distr = GradNormal(mu, sigma)
            # rpeps_actions_batch = distr.eps_to_action(rpeps_batch)
            # alpha_actions_batch = input_dict['alpha_policy_actions']
            
            # rp_a_loss = torch.norm(rpeps_actions_batch - alpha_actions_batch, p=2, dim=-1)
            # rp_a_loss = rp_a_loss.mean()
            
            blending = 0.
            a_loss = (a_loss * blending) + (rp_a_loss * (1.0 - blending))
            
            loss = a_loss + 0.5 * c_loss * self.critic_coef # - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)
