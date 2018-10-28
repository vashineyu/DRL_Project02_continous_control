import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pip
pip.main(['-q','install', './python', '--user'])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', default = 200, type = int)
parser.add_argument('--use_gpu', default = 0)
FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if FLAGS.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
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
