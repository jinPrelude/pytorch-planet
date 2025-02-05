# References
#       PlaNet Paper: https://arxiv.org/pdf/1811.04551
#
#       Implementation:
#           1. Danijar's repo: [https://github.com/danijar/planet]
#           1. Jaesik's repo: [https://github.com/jsikyoon/dreamer-torch]
#           2. Kaito's repo: [https://github.com/cross32768/PlaNet_PyTorch]

import torch
from torch import nn
from torch.distributions import kl_divergence
from networks import EncoderModel, RepresentationModel, RecurrentModel, TransitionModel, DecoderModel, RewardModel
from utils import get_device, get_dtype


class Planet(nn.Module):
    def __init__(self, params, action_dim):
        super(Planet, self).__init__()
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        self.action_dim = action_dim
        self.rnn_model = RecurrentModel(params=self.params, action_dim=self.action_dim) # 그냥 hidden_state + action 같이 받는 GRU
        self.obs_encoder = EncoderModel(params=self.params) # 그냥 CNN
        self.repr_model = RepresentationModel(params=self.params) # VAE 처럼 mu, sigma 출력. impl. details 있으니 들어가보셈.
        self.transition_model = TransitionModel(params=self.params) # VAE 처럼 mu, sigma 출력. impl. details 있으니 들어가보셈.
        self.decoder_model = DecoderModel(params=self.params)
        self.reward_model = RewardModel(params=self.params)

    def __repr__(self):
        return 'PlaNet'

    def get_init_h_state(self, batch_size): # GRU hidden state 초기화
        return torch.zeros((batch_size, self.params['h_dim']), dtype=self.d_type, device=self.device)

    def forward(self, sampled_episodes):
        dist_predicted = {'prior': list(), 'posterior': list(), 'recon_obs': list(), 'reward': list()}
        h_state = self.get_init_h_state(batch_size=self.params['batch_size']) # (batch_size, h_dim)
        # sequence iteration 시작 (batch_wise가 아닌 sequence_wise)
        for time_stamp in range(self.params['chunk_length']):
            input_obs = sampled_episodes['obs'][time_stamp] # 모든 batch의 첫번째 obs
            noisy_input_obs = (1/pow(2, self.params['pixel_bit']))*torch.randn_like(input_obs) + input_obs # T_T.md 12번. generalization 성능 향상을 위한 noise 추가
            action = sampled_episodes['action'][time_stamp]

            encoded_obs = self.obs_encoder(noisy_input_obs) # embed size: 1024
            z_prior = self.transition_model(h_state) # MLP. mu, sigma 출력 후 dist 객체 출력
            z_posterior = self.repr_model(h_state, encoded_obs) # MLP. mu, sigma 출력 후 dist 객체 출력

            z_state = z_posterior.rsample() # reparameterization sampling

            dist_recon_obs = self.decoder_model(h_state, z_state)
            dist_reward = self.reward_model(h_state, z_state) # MLP. mu, sigma 출력 후 dist 객체 출력
            h_state = self.rnn_model(h_state, z_state, action)

            dist_predicted['prior'].append(z_prior)
            dist_predicted['posterior'].append(z_posterior)
            dist_predicted['recon_obs'].append(dist_recon_obs)
            dist_predicted['reward'].append(dist_reward)
        return dist_predicted

    def compute_loss(self, target, dist_predicted):
        sampled_reconstructed_obs = torch.stack([dist_recon_obs.rsample() for dist_recon_obs in dist_predicted['recon_obs']])
        sampled_reward = torch.stack([dist_reward.rsample() for dist_reward in dist_predicted['reward']]) # reward도 sampling 하네.
        # Individual loss terms
        recon_loss = ((target['obs'] - sampled_reconstructed_obs) ** 2).mean(dim=0).mean(dim=0).sum() # T_T.md 4번.
        kl_loss = torch.stack(
            [kl_divergence(p=dist_posterior, q=dist_prior) for dist_prior, dist_posterior in
             zip(dist_predicted['prior'], dist_predicted['posterior'])]
        ) # T_T.md 5번. kl 거리에 순서가 중요한 줄 몰랐다. / 해보니까 바꿔도 학습 되긴 함.
        kl_loss = (torch.maximum(kl_loss, torch.tensor([self.params['free_nats']])[0])).mean() # A.Hyper parameters. free_nats가 뭔데?
        reward_prediction_loss = ((target['reward'] - sampled_reward) ** 2).mean()
        # Net loss term
        net_loss = recon_loss + kl_loss + reward_prediction_loss
        return net_loss, (recon_loss.item(), kl_loss.item(), reward_prediction_loss.item())

