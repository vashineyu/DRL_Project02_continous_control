from unityagents import UnityEnvironment
import numpy as np

import argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', default = 1000, type = int)
parser.add_argument('--experiment_tag', default = None, type = str)
parser.add_argument('--use_noise', default = 1)
FLAGS = parser.parse_args()

# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='../Reacher_Linux_NoVis/Reacher.x86_64')
#env = UnityEnvironment(file_name='../Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

############################
from agent import Agent
from collections import deque
import torch

agent = Agent(state_size=state_size, action_size=action_size, random_seed=np.random.randint(10000))
def ddpg(n_episodes=1000, max_t=2000):
    scores_deque = deque(maxlen=100)
    average_scores = []                                        # average of the score of the 20 agents for each episode
        
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        
        for t in range(max_t):
            actions = agent.act(states, add_noise = FLAGS.use_noise)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states                               # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)            
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        score = np.mean(scores)
        scores_deque.append(score)
        average_scores.append(score)      
        
        if i_episode % 10 == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}'\
                .format(i_episode, np.mean(scores_deque), np.max(scores), np.min(scores)), end="\n")  
            
        if np.mean(scores_deque) >= 30.0:
            torch.save(agent.actor_local.state_dict(), 'actor.pth')
            torch.save(agent.critic_local.state_dict(), 'critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))            
            break            
            
    return average_scores

scores = ddpg(n_episodes = FLAGS.num_episodes)
# Plot the Result #
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize = (6,4))
plt.plot(np.arange(1, len(scores)+1), scores)
plt.title("Result")
plt.xlabel('Episode', fontsize = 16)
plt.ylabel('Average Scores', fontsize = 16)
plt.tight_layout()
if FLAGS.experiment_tag is not None:
    sav_name = "result_" + FLAGS.experiment_tag + ".png"
else:
    sav_name = "result.png"
plt.savefig(sav_name)