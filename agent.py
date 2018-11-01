import numpy as np
import os
import tensorflow as tf

"""
Implementation of DDPG for solving continous control
"""

class Actor():
    def __init__(self, session, state_size, action_size, action_bound = 1, learning_rate = 0.001, tau = 0.001, batch_size = 64):
        """
        Actor network
        Args:
        - state_size: size of states
        - action_size: size of available actions
        - action_bound: the real continous action will be range from "-action_bound" to "+action_bound"
        - learning_rate: optimizer lr
        
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.lr = learning_rate
        self.bz = batch_size
        
        print("State size: %i" % self.state_size)
        print("Action size: %i" % self.action_size)
        
        # --- Init Model --- #
        self.sess = session
        
        self.state = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32)
        with tf.variable_scope('local_net'):
            self.out, self.scaled_out = self._build_actor_network()
            
        with tf.variable_scope('target_net'):
            self.target_out, self.target_scaled_out = self._build_actor_network()
            
        self.localnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'local_net')
        self.targetnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        self.params_replace = [tf.assign(old * tau, new * (1.-tau) ) for old, new in zip(self.targetnet_params, self.localnet_params)]
        
        self.action_gradient = tf.placeholder(shape = [None, self.action_size], dtype = tf.float32)
        self.unnorm_actor_gradients = tf.gradients(self.out, self.localnet_params, -self.action_gradient)
        self.action_gradients = list(map(lambda x: tf.div(x, self.bz), self.unnorm_actor_gradients))
        
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.action_gradients, self.localnet_params))
        
        # --- Checkpoint and summary #
        self.saver = tf.train.Saver()
        
    def predict(self, inputs):
        value = self.sess.run(self.scaled_out, feed_dict = {self.state:inputs})
        return value
    
    def predtict_target(self):
        value = self.sess.run(self.target_scaled_out, feed_dict = {self.state:inputs})
        return value
    
    def train(self, inputs, this_gradient):
        # No return function
        _ = self.sess.run(self.train_op, feed_dict = {self.state: inputs, self.action_gradient: this_gradient})
        
    def update_target_network(self):
        # No return fuction
        self.sess.run(params_replace)
        
    def _build_actor_network(self, neurons_per_layer = [256, 256, 128]):
        # --- Actor ---  #
        # Policy network #
        def mlp_block(x, units):
            x = tf.layer.dense(x, units)
            x = tf.nn.relu(x)
            return x
        
        for i, n in enumerate(neurons_per_layer):
            if i == 0:
                x = mlp_block(self.state, n)
            else:
                x = mlp_block(x, n)
        
        out = tf.layer.dense(x, self.action_size)
        scale_out = tf.nn.tanh(x) * self.action_bound
        
        return out, scale_out
        
class Critic():
    def __init__(self):
        pass
    
    def predict(self):
        pass
    
    def predict_target(self):
        pass
    
    def train(self):
        pass
    
    def action_gradients(self):
        pass
    
    def update_target_network(self):
        pass
    
    def _build_critic_network(self):
        pass
    
    
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    
    fake_session = tf.Session()
    fake_state_size = 33
    fake_action_size = 4
    actor = Actor(session = fake_session, state_size = fake_state_size, action_size = fake_action_size)
    print("Build Actor Pass")
    
    
    
    