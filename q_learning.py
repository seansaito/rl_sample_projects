import gym
import numpy as np

# Create the environment
game_name = 'FrozenLake-v0'
env = gym.make(game_name)

def q_learning(learning_rate, y, num_episodes):
    #Initialize table with all zeros
    q_table = np.zeros([env.observation_space.n,env.action_space.n])
    rewards = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        all_rewards = 0
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            # Epsilon-Greedy action selection
            a = np.argmax(q_table[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            s1,r,d,_ = env.step(a)
            # Update the table
            q_table[s,a] = q_table[s,a] + learning_rate * (r + y * np.max(q_table[s1,:]) - q_table[s,a])
            all_rewards += r
            s = s1
            if d == True:
                break
        rewards.append(all_rewards)

