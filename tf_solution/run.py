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
parser.add_argument('--num_episodes', default = 250, type = int)
parser.add_argument('--batch_size', default = 64, type = int)
parser.add_argument('--buffer_size', default = 1000, type = int)
parser.add_argument('--use_gpu', default = 0)
parser.add_argument('--experiment_tag', default = None, type = str)
parser.add_argument('--use_noise', default = 1)
FLAGS = parser.parse_args()

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
from agent import Actor, Critic, OrnsteinUhlenbeckActionNoise, build_summary
from utils import ReplayBuffer

def train(sess, env, FLAGS, actor, critic, actor_noise):
    time_steps = 20
    num_update = 10
    t_max = 800
    sess.run(tf.global_variables_initializer())    
    
    avg_score = [] # record agents' mean scores over episodes
    scores_deque = deque(maxlen = 100) # smoothed average scores
    
    len_agents = len(str(num_agents))
    
    env_info  = env.reset(train_mode=True)[brain_name]
    # Make local and target network have same initalized weights
    actor.update_target_network()
    critic.update_target_network()
    
    replay_buffer = ReplayBuffer(FLAGS.buffer_size)
    
    for i_episode in range(FLAGS.num_episodes):
        env_info  = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        
        # Scroe reset for the episode
        scores = np.zeros(num_agents)
        
        counter = 0
        while True:
            counter += 1
            # Generate action by Actor's local_network
            noise = actor_noise() if FLAGS.use_noise else 0.
            action = actor.predict(np.reshape(state, (1, actor.state_size))) + noise

            env_info = env.step(action[0])[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            replay_buffer.add(np.reshape(state, (actor.state_size,)), 
                              np.reshape(action, (actor.action_size,)),
                              reward,
                              done,
                              np.reshape(next_state, (actor.state_size,))
                             )

            if (counter % time_steps == 0):
                for _ in range(num_update):
                    s1_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(FLAGS.batch_size)
                    a2_batch = actor.predict_target(s2_batch)
                    q_target_next = critic.predict_target(s2_batch, a2_batch)
                    # reward + (not done * gamma * q_target_next)
                    q_target = r_batch + (1.-t_batch) * 0.9 * q_target_next.ravel()
                    # Evaluate Actors' action by critic and train the critic  
                    current_q_value = critic.train(states = s1_batch,
                                                   actions = a_batch,
                                                   q_targets = q_target)

                    a_outs = actor.predict(s1_batch)
                    grads = critic.action_gradients(s1_batch, a_outs)

                    # train actor
                    actor.train(s1_batch, grads)

                    actor.update_target_network()
                    critic.update_target_network()
            
            state = next_state
            scores += reward
            
            if np.any(done):
                break

        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        
        print('\rEpisode {}\t Episode score:{:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        actor.save_model()
        critic.save_model()
        
        
    return avg_score

# Run it
with tf.Session() as sess:
    action_bound = 1
    actor = Actor(sess, state_size, action_size, action_bound, batch_size = FLAGS.batch_size)
    critic = Critic(sess, state_size, action_size, batch_size = FLAGS.batch_size)
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))
    
    scores = train(sess, env, FLAGS, actor, critic, actor_noise)
    
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