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
parser.add_argument('--num_episodes', default = 200, type = int)
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--buffer_size', default = 100000, type = int)
parser.add_argument('--n_to_soft_update', default = 4, type = int)
parser.add_argument('--use_gpu', default = 0)
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
env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

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
    
    summary_ops, summary_vars = build_summary()
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter("./record", sess.graph)
    
    # Make local and target network have same initalized weights
    actor.update_target_network()
    critic.update_target_network()
    
    replay_buffer = ReplayBuffer(FLAGS.buffer_size)
    
    for i in tqdm(range(FLAGS.num_episodes)):
        env_info  = env.reset(train_mode=True)[brain_name]
        counter = 0
        ep_reward = 0
        ep_ave_max_q = 0
        
        while True:
            counter += 1
            state = env_info.vector_observations[0]
            
            #state = np.reshape(state, (1, 33))
            
            action = actor.predict(np.reshape(state, (1, actor.state_size))) + actor_noise()

            env_info = env.step(action[0])[brain_name]
            next_state = env_info.vector_observations   # get the next state
            reward = env_info.rewards                   # get the reward
            done = env_info.local_done                  # see if episode has finished

            replay_buffer.add(np.reshape(state, (actor.state_size,)), 
                              np.reshape(action, (actor.action_size,)),
                              reward,
                              done,
                              np.reshape(next_state, (actor.state_size,))
                             )

            if replay_buffer.size() > FLAGS.buffer_size:
                s1_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(FLAGS.batch_size)

                target_q = critic.predict_target(s2_batch, 
                                                 actor.predict_target(s2_batch))

                y_i = []
                for k in range(FLAGS.batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # train critic and get predicted_q_value
                predicted_q_value = critic.train(s_batch, a_batch, np.reshape(y_i, (FLAGS.batch_size, 1) ))
                ep_ave_max_q += np.amax(predicted_q_value)

                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)

                # train actor
                actor.train(s_batch, grads)

                if FLAGS.n_to_soft_update % counter == 0:
                    actor.update_target_network()
                    critic.update_target_network()
                
                state = next_state
                ep_reward += r
            
            if np.any(done):
                s_value = sess.run(summary_ops, feed_dict = {summary_vars[0]: ep_reward, summary_vars[1]: ep_ave_max_q / float(counter)})
                writer.add_summary(s_value, i)
                writer.flush()
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(counter))))
                break

# Run it
with tf.Session() as sess:
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_bound = 1
    
    actor = Actor(sess, state_size, action_size, action_bound)
    critic = Critic(sess, state_size, action_size)
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size))
    
    train(sess, env, FLAGS, actor, critic, actor_noise)
    
print("Process Done")
    