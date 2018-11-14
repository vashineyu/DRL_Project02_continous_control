import numpy as np
import os
import tensorflow as tf

"""
Implementation of DDPG for solving continous control
"""

class Actor():
    def __init__(self, session, state_size, action_size, action_bound = 1, learning_rate = 1e-3, tau = 1e-2, batch_size = 64):
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
        self.learning_phase = tf.placeholder(dtype = tf.bool, name = 'phase')
        
        with tf.variable_scope("Actor"):
            with tf.variable_scope('local_net'):
                self.out, self.scaled_out = self._build_actor_network(trainable = True)
            with tf.variable_scope('target_net'):
                self.target_out, self.target_scaled_out = self._build_actor_network(trainable = False)
            
        self.localnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/local_net')
        self.targetnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/target_net')
        
        self.params_replace = [tf.assign(old, old * (1.-tau) + tau * new) for old, new in zip(self.targetnet_params, self.localnet_params)]
        
        #optimizer = tf.train.RMSPropOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.action_gradient = tf.placeholder(shape = [None, self.action_size], dtype = tf.float32)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.unnorm_actor_gradients = tf.gradients(self.out, self.localnet_params, -self.action_gradient)
            self.action_gradients = list(map(lambda x: tf.div(x, self.bz), self.unnorm_actor_gradients))
            self.train_op = optimizer.apply_gradients(zip(self.action_gradients, self.localnet_params))
        
        # --- Checkpoint and summary #
        self.saver = tf.train.Saver()
        
    def predict(self, inputs):
        value = self.sess.run(self.scaled_out, feed_dict = {self.state:inputs, 
                                                            self.learning_phase: False})
        return value
    
    def predict_target(self, inputs):
        value = self.sess.run(self.target_scaled_out, feed_dict = {self.state:inputs, 
                                                                   self.learning_phase: False})
        return value
    
    def train(self, inputs, this_gradient):
        # No return function
        _ = self.sess.run(self.train_op, feed_dict = {self.state: inputs, 
                                                      self.action_gradient: this_gradient,
                                                      self.learning_phase: True})
        
    def update_target_network(self):
        # No return fuction
        self.sess.run(self.params_replace)
        
    def _build_actor_network(self, neurons_per_layer = [256, 256], trainable = True):
        # --- Actor ---  #
        # Policy network #
        def mlp_block(x, units, trainable = trainable):
            x = tf.layers.dense(x, units, trainable = trainable)
            x = tf.layers.batch_normalization(x, trainable = trainable, training = self.learning_phase)
            x = tf.nn.relu(x)
            return x
        
        for i, n in enumerate(neurons_per_layer):
            if i == 0:
                x = mlp_block(self.state, n)
            else:
                x = mlp_block(x, n)
        
        out = tf.layers.dense(x, units = self.action_size, trainable = trainable)
        scale_out = tf.nn.tanh(out) * self.action_bound
        
        return out, scale_out
    
    def save_model(self, model_name = None):
        self.saver.save(self.sess, './actor.ckpt' if model_name is None else model_name)
        print("Model saved!")
        
    def load_model(self, model_name = None):
        self.saver.restore(self.sess, './actor.ckpt' if model_name is None else model_name)
        print("Model loaded!")
        
class Critic():
    def __init__(self, session, state_size, action_size, learning_rate = 1e-3, tau=1e-2, gamma=0.9, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # init the model #
        self.sess = session
        global_step = tf.Variable(0, trainable=False)
        self.state = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32)
        self.next_state = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32)
        self.action = tf.placeholder(shape = [None, self.action_size], dtype = tf.float32)
        
        self.learning_phase = tf.placeholder(dtype = tf.bool, name = 'phase')
        
        with tf.variable_scope("Critic"):
            self.local_out = self._build_critic_network(self.state, self.action, 
                                                        scope = 'local', trainable = True)
            self.target_out = self._build_critic_network(self.next_state, self.action, 
                                                         scope = 'target', trainable = False)
                
        # Handlers for parameters
        self.localnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/local')
        self.targetnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/target')
        self.params_replace = [tf.assign(old, old * (1.-tau) + tau * new) for old, new in zip(self.targetnet_params, self.localnet_params)]

        
        # Compute loss for critic network
        self.q_target = tf.placeholder(shape = [None], dtype = tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.q_target - self.local_out))
        
        
        lr_c = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.999)
        
        #optimizer = tf.train.RMSPropOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(lr_c)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #self.train_op = optimizer.minimize(self.loss)
            c_grads = optimizer.compute_gradients(self.loss, var_list = self.localnet_params)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in c_grads]
            self.train_op = optimizer.apply_gradients(capped_gvs)
        
        # self.local_out should be single value
        self.action_gradient = tf.gradients(ys = self.local_out, xs = self.action)[0]
        
        self.saver = tf.train.Saver()
        
    
    def predict(self, inputs, action):
        out = self.sess.run(self.local_out, feed_dict = {self.state: inputs, 
                                                         self.action: action,
                                                         self.learning_phase: False})
        return out
    
    def predict_target(self, inputs, action):
        out = self.sess.run(self.target_out, feed_dict = {self.next_state: inputs, 
                                                          self.action: action,
                                                          self.learning_phase: False})
        return out
    
    def train(self, states, actions, q_targets):
        """
        Train critic and return current q-value
        """
        
        out, _ = self.sess.run([self.local_out, self.train_op], 
                               feed_dict = {self.state: states, 
                                            self.action: actions,
                                            self.q_target: q_targets,
                                            self.learning_phase: True})
        return out
    
    def action_gradients(self, inputs, actions):
        out = self.sess.run(self.action_gradient, feed_dict = {self.state: inputs, 
                                                               self.action: actions,
                                                               self.learning_phase: False})
        return out
    
    def update_target_network(self):
        self.sess.run(self.params_replace)
    
    def _build_critic_network(self, input_state, input_action, 
                              scope, neurons_per_layer = [256, 256], trainable = True):
        # --- Critic, Q-network --- #
        def mlp_block(x, units, trainable):
            x = tf.layers.dense(x, units, trainable = trainable)
            x = tf.layers.batch_normalization(x, trainable = trainable, training = self.learning_phase)
            x = tf.nn.relu(x)
            return x
        # ------------------------- #
        with tf.variable_scope(scope):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.state_size, n_l1], trainable = trainable)
            w1_a = tf.get_variable('w1_a', [self.action_size, n_l1], trainable = trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable = trainable)
            x = tf.matmul(input_state, w1_s) + tf.matmul(input_action, w1_a) + b1
            x = tf.layers.batch_normalization(x, trainable = trainable, training = self.learning_phase)
            x = tf.nn.relu(x)
                      
        for i, n in enumerate(neurons_per_layer):
            x = mlp_block(x, n, trainable)
        
        out = tf.layers.dense(x, 1, trainable = trainable)
        return out
    
    def save_model(self, model_name = None):
        self.saver.save(self.sess, 'critic.ckpt' if model_name is None else model_name)
        print("Model saved!")
        
    def load_model(self, model_name = None):
        self.saver.restore(self.sess, 'critic.ckpt' if model_name is None else model_name)
        print("Model loaded!")
    
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    
def build_summary():
    episode_reward = tf.Variable(0.)
    episode_ave_max_q = tf.Variable(0.)
    
    tf.summary.scalar('Reward', episode_reward)
    tf.summary.scalar('Qmax_value', episode_ave_max_q)
    merge_ops = tf.summary.merge_all()
    
    return merge_ops, [episode_reward, episode_ave_max_q]
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    
    fake_session = tf.Session()
    fake_state_size = 33
    fake_action_size = 4
    actor = Actor(session = fake_session, state_size = fake_state_size, action_size = fake_action_size)
    print("Build Actor Pass")
    
    critic = Critic(session = fake_session, state_size = fake_state_size, action_size = fake_action_size)
    print("Build Critic Pass")
    
    
    
    