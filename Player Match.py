from time import sleep
from model import SAC


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from pettingzoo.classic import connect_four_v3

import pygame

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device.type

env = connect_four_v3.env(render_mode="human")
#env = connect_four_v3.env()
env.reset()


#Numero de iteraciones
num_episodes = 2000

# Definicion de variables
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.0001
TAU = 0.0005
LR = 3e-4

# Get number of actions from gym action space
n_actions = 7

# Get the number of state observations
observation, reward, termination, truncation, info = env.last()

#observation_shape = np.array(observation["observation"]).reshape(1,-1)
n_observations = 84


# Creamos las 3 instancias

sac_agent_0 = SAC(name="agent_0", lr=LR,gamma=GAMMA,tau=TAU,
                eps_start=EPS_START,eps_end=EPS_END,eps_dec=EPS_DECAY,
                n_observations=n_observations,n_actions=n_actions)

sac_agent_1 = SAC(name="agent_1", lr=LR,gamma=GAMMA,tau=TAU,
                eps_start=EPS_START,eps_end=EPS_END,eps_dec=EPS_DECAY,
                n_observations=n_observations,n_actions=n_actions)

sac_agent_a = SAC(name="agent_a", lr=LR,gamma=GAMMA,tau=TAU,
                eps_start=EPS_START,eps_end=EPS_END,eps_dec=EPS_DECAY,
                n_observations=n_observations,n_actions=n_actions)


##Cargamos los agentes entrenados

sac_agent_0.load_model()
sac_agent_1.load_model()
sac_agent_a.load_model()

# Creamos el primer entorno

agent_0_score = []
agent_a_score = []



total_reward_player_0 = 0
total_reward_player_a = 0
last_action_player_0 = 0
last_action_player_a = 0

env.reset()
observation, reward, termination, truncation, info = env.last()
previous_state = observation

for episode in range(num_episodes):
    state = env.reset()
    
    #Definicion del primer jugador en cada partida
    if(episode % 2 == 1):
        for agent in env.agent_iter():
            
            
            state, reward, termination, truncation, info = env.last()


            if termination or truncation:
                    # Registramos las recompensas
                    total_reward_player_0 = total_reward_player_0 + env.rewards["player_0"]
                    total_reward_player_a = total_reward_player_a + env.rewards["player_1"]

                    agent_0_score.append(total_reward_player_0)
                    agent_a_score.append(total_reward_player_a)

                    break
            
            if(agent == "player_0"):
                action = sac_agent_0.choose_action(agent,state,env)
                env.step(action)
            else:
                action = (int(input("Elige un movimiento del 1-7: ")) - 1)
                env.step(action) 
            previous_state = state
        env.close()
    else:
        for agent in env.agent_iter():
            #print(agent)
            state, reward, termination, truncation, info = env.last()


            if termination or truncation:
                    #print(env.rewards)
                    total_reward_player_0 = total_reward_player_0 + env.rewards["player_1"]
                    total_reward_player_a = total_reward_player_a + env.rewards["player_0"]

                    agent_0_score.append(total_reward_player_0)
                    agent_a_score.append(total_reward_player_a)

                    break
            
            if(agent == "player_1"):
                action = sac_agent_0.choose_action(agent,state,env)
                env.step(action)
            else:
                action = (int(input("Elige un movimiento del 1-7: ")) - 1)
                env.step(action) 
            previous_state = state
        env.close()

    sleep(0.4)


print("Total score Competitive SAC Agent: " + str(total_reward_player_0))
print("Total score Alone SAC Agent: " + str(total_reward_player_a))