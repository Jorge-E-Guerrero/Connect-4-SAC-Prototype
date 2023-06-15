from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode='human')
env.reset()
env.render()

print(env.agent_iter)


num_episodes = 20
total_reward_player_0 = 0
total_reward_player_1 = 0

for episode in range(num_episodes):


    state = env.reset()
    env.render()
    


    for agent in env.agent_iter():
        #print(agent)
        observation, reward, termination, truncation, info = env.last()


        if termination or truncation:
                print(env.rewards)
                total_reward_player_0 = total_reward_player_0 + env.rewards["player_0"]
                total_reward_player_1 = total_reward_player_1 + env.rewards["player_1"]
                break
        
        if(agent == "player_0"):
            #print("Player 1")

            mask = observation["action_mask"]
            action = env.action_space(agent).sample(mask)  # this is where you would insert your policy
            #print(action)
            env.step(action)  
        else:
            #print("Player 2")
            
            mask = observation["action_mask"]
            #print(mask)
            action = env.action_space(agent).sample(mask)  # this is where you would insert your policy
            #print(action)
            #action = int(input("Elige la columna: \n")) - 1 

            env.step(action)  
    env.close()

print("Total score player 1: " + str(total_reward_player_0))
print("Total score player 2: " + str(total_reward_player_1))