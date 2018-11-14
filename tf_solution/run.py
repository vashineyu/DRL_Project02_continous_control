import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', default = 0)
parser.add_argument('--experiment_tag', default = None, type = str)
FLAGS = parser.parse_args()

MAX_EPISODES = 1200
UPDATE_PER_ITER = 20
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 10 #1100
REPLACE_ITER_C = 10 #1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if FLAGS.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import tensorflow as tf

# -------------------------- #
from unityagents import UnityEnvironment
print("Start to load unity ENV")
env = UnityEnvironment(file_name='../Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# get the default brain
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

### Build Training loop
# each epoch have 1000 steps, per contact get 0.1 reward
from agent import Actor, Critic, OrnsteinUhlenbeckActionNoise, build_summary, Memory
from utils import ReplayBuffer

def train(sess, env, actor, critic, actor_noise, M):
    t_max = 2000
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())    
    avg_score = []
    scores_deque = deque(maxlen = 100)
    len_agents = len(str(num_agents))
    
    for i_episode in range(1, MAX_EPISODES+1):
        scores = np.zeros(num_agents)
        env_info  = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations[0]
        actor_noise.reset()
        
        for counter in range(t_max):       
            # Generate action by Actor's local_network
            actions = np.clip(actor.act(states) + actor_noise(), -1, 1) #
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations[0]   # get the next state
            rewards = env_info.rewards[0]                   # get the reward
            dones = env_info.local_done[0]                  # see if episode has finished

            M.store_transition(states, actions, rewards, next_states)
            if (counter % UPDATE_PER_ITER == 0) & (M.pointer > MEMORY_CAPACITY):
                for _ in range(10):
                    b_M = M.sample(BATCH_SIZE)
                    b_s = b_M[:, :STATE_DIM]
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                    b_s_ = b_M[:, -STATE_DIM:]

                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)
            
            states = next_states
            scores += rewards
            
            if np.any(dones):
                break
        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        
        print('\rEpisode {}\tEpisode Score: {:.2f}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque), np.max(avg_score)), end="")
        
        if np.mean(scores_deque) >= 30.:
            print("Game solved")
            saver.save(sess, "./ddpg.ckpt", write_meta_graph = False)
            break
    return avg_score

# Run it
tf.reset_default_graph()
with tf.Session() as sess:
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = ACTION_DIM = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = STATE_DIM = states.shape[1]
    action_bound = ACTION_BOUND = 1
    print("State size: %i, Action size: %i, Action bound: %.2f" % (STATE_DIM, ACTION_DIM, ACTION_BOUND) )
    """
    Set model
    """
    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, state_size], name='s')
    
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
        
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, state_size], name='s_')
    
    actor = Actor(sess, action_size, action_bound, learning_rate=LR_A, t_replace_iter=REPLACE_ITER_A, S = S, R = R, S_ = S_)
    critic = Critic(sess, state_size, action_size, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_, S= S, R = R, S_ = S_)
    actor.add_grad_to_graph(critic.a_grads)
    M = Memory(MEMORY_CAPACITY, dims=2 * state_size + action_size + 1)
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))
    
    scores = train(sess, env, actor, critic, actor_noise, M)
    
print("")
print("Process Done")
    
# Plot the Result #
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize = (8,6))
plt.plot(np.arange(1, len(scores)+1), scores)
plt.title("Result")
plt.xlabel('Episode', fontsize = 16)
plt.ylabel('Average Scores', fontsize = 16)
if FLAGS.experiment_tag is not None:
    sav_name = "result_" + FLAGS.experiment_tag + ".png"
    df_name = "result_" + FLAGS.experiment_tag + ".csv"
else:
    sav_name = "result.png"
    df_name = "result.csv"
plt.savefig(sav_name)

df = pd.DataFrame({'score_of_episode': scores})
df.to_csv(df_name)