import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from pettingzoo.classic import connect_four_v3
from pettingzoo.utils.env import ActionType, AECEnv, AECIterable, AECIterator, ObsType
from pettingzoo.utils.env_logger import EnvLogger
from pettingzoo.utils.wrappers.base import BaseWrapper

# Define la red neuronal para el actor y el crítico
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = torch.softmax(self.fc3(x), dim=-1)
        critic_output = self.fc3(x)
        return actor_output, critic_output

# Define el algoritmo Soft Actor-Critic
class SAC:
    def __init__(self, env, actor_critic, optimizer):
        self.env = env
        self.actor_critic = actor_critic
        self.optimizer = optimizer

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        actor_output, _ = self.actor_critic(state)
        action_probs = actor_output.detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.env.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).view(-1, 1)

        actor_output, critic_output = self.actor_critic(states)
        _, next_critic_output = self.actor_critic(next_states)

        q_values = critic_output.gather(1, actions)
        next_q_values = next_critic_output.max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * 0.99 * next_q_values

        critic_loss = F.mse_loss(q_values, expected_q_values.detach())
        actor_loss = torch.mean(actor_output.log() * (q_values - critic_output.detach()))

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()

# Crea el entorno Connect-4-v3
env = connect_four_v3.env()

env.reset()
#env.render()

#print(env.observation_space(agent).shape)

#print(np.ndim(env.observation_spaces["player_0"]["observation"])) 



# Obtén el tamaño de entrada y de salida del entorno
input_size = np.ndim(env.observation_spaces["player_0"]["observation"])

print(env.action_spaces)
output_size = 7
"""
# Crea la red neuronal del actor y el crítico
actor_critic = ActorCritic(input_size, output_size)

# Crea el optimizador para el algoritmo Soft Actor-Critic
optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)

# Crea una instancia de SAC
sac = SAC(env, actor_critic, optimizer)

# Entrenamiento
num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = sac
"""