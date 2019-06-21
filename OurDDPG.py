import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(torch.cat([x, u], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x 


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, actor_lr, is_ro):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.is_ro = is_ro


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			if self.is_ro:
				actions = self.actor(state)
				repeat_number = 128
				# augmented_actions_raw = actions.repeat(1, repeat_number).reshape([batch_size * repeat_number, -1])
				augmented_states = state.repeat(1, repeat_number).reshape([batch_size * repeat_number, -1])

				noises = []
				for batch_i in range(batch_size):
					mins = torch.max(actions[batch_i] - 0.1, (-torch.ones([actions.shape[1]])).cuda())
					maxs = torch.min(actions[batch_i] + 0.1, torch.ones([actions.shape[1]]).cuda())
					noises.append(torch.rand(repeat_number - 1, actions.shape[1]).cuda() * (maxs - mins) + mins)
					noises.append(actions[batch_i:batch_i + 1])
				noises = torch.cat(noises, 0).cuda()
				augmented_actions = noises
				# augmented_actions = torch.clamp(augmented_actions, -1, 1)

				q_value = self.critic(augmented_states, augmented_actions)[:, 0]
				split_q_value = torch.split(q_value, repeat_number, dim=0)
				split_action = torch.split(augmented_actions, repeat_number, dim=0)

				actor_loss_ro = 0
				for batch_i in range(batch_size):
					max_index = torch.argmax(split_q_value[batch_i])
					target_action = split_action[batch_i][max_index, :].detach()
					actor_loss_ro = actor_loss_ro + F.mse_loss(actions[batch_i, :], target_action)
			else:
				actor_loss_dpg = -self.critic(state, self.actor(state)).mean()

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			if self.is_ro:
				actor_loss_ro.backward()
			else:
				actor_loss_dpg.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
