# Use A3C (Asynchronous Advantage Actor-Critic) to solve Atari Games
# A3C implementation with GPU optimizer threads


from __future__ import print_function
from __future__ import division
from builtins import input

import gym
import time
import threading
import numpy as np
import tensorflow as tf
from scipy.misc import imresize


# ------------------ Constants ------------------
GAME = 'SpaceInvaders-v0'

TRAIN_TIME = 60 # seconds

THREAD_DELAY = 0.001 # seconds
NUM_ENV_THREADS = 8 # number of environment threads
NUM_OPT_THREADS = 2 # number of optimizer threads

IMAGE_WIDTH  = 84
IMAGE_HEIGHT = 84
IMAGE_STACK  = 4
NONE_STATE   = np.zeros((IMAGE_STACK, IMAGE_HEIGHT, IMAGE_WIDTH))

GAMMA = 0.99 # discount factor

N_STEP_RETURN = 2 # can try 4 or 8 or any other n-step else
GAMMA_N = GAMMA ** N_STEP_RETURN

MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
EXPLORATION_STOP = 100000 # at this step epsilon will be 0.1
EPSILON_DECAY    = (MIN_EPSILON - MAX_EPSILON) / EXPLORATION_STOP

MIN_BATCH_SZ = 32
MAX_BATCH_SZ = MIN_BATCH_SZ * 5

LEARNING_RATE = 0.00025
DECAY = 0.99

LOSS_VALUE   = 0.5  # value loss coefficient
LOSS_ENTROPY = 0.01 # entropy coefficient


# ------------------ Utilities ------------------
def process_image(img):
    rgb = imresize(img, size=(IMAGE_HEIGHT, IMAGE_WIDTH), interp='nearest')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b # the effective luminance of a pixel

    out = gray.astype(np.float32) / 127.5 - 1 # normalize
    return out


# ------------------ Classes ------------------
class Brain:

    def __init__(self, num_state, num_actions):
        self.train_queue = [ [], [], [], [], [] ] # s, a, r, s', s'_terminal_mask
        self.lock_queue  = threading.Lock()

        input_shape = (None, num_state[0], num_state[1], num_state[2])

        # placeholders
        self.states  = tf.placeholder(tf.float32, shape=input_shape, name='states' )
        self.actions = tf.placeholder(tf.int32,   shape=(None, )   , name='actions')
        self.returns = tf.placeholder(tf.float32, shape=(None, )   , name='returns') # discounted n-step return

        # build the graph
        Z = tf.transpose(self.states, [0, 2, 3, 1])
        Z = tf.contrib.layers.conv2d(Z, 16, (8, 8), stride=4, activation_fn=tf.nn.relu)
        Z = tf.contrib.layers.conv2d(Z, 32, (4, 4), stride=2, activation_fn=tf.nn.relu)
        Z = tf.contrib.layers.flatten(Z)
        Z = tf.contrib.layers.fully_connected(Z, 256, activation_fn=tf.nn.relu)

        out_policy = tf.contrib.layers.fully_connected(Z, num_actions, activation_fn=tf.nn.softmax)
        value      = tf.contrib.layers.fully_connected(Z, 1, activation_fn=lambda x: x)
        out_value  = tf.reshape(value, [-1])

        self.predict_p = out_policy
        self.predict_v = out_value

        # calculate the loss
        selected_action_prob = tf.reduce_sum(out_policy * tf.one_hot(self.actions, num_actions), axis=1)
        log_prob = tf.log(selected_action_prob + 1e-10)
        advantage = self.returns - out_value

        loss_policy  = - log_prob * tf.stop_gradient(advantage) # maximize policy performance
        loss_value   = LOSS_VALUE * tf.square(advantage)       # minimize value error
        loss_entropy = LOSS_ENTROPY * tf.reduce_sum(out_policy * tf.log(out_policy + 1e-10), axis=1) # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + loss_entropy)

        self.train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY).minimize(loss_total)

        # set the session and initialize variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # avoid modifications
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def predict_policy(self, states):
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=0)
        return self.session.run(self.predict_p, feed_dict={self.states: states})

    def predict_value(self, states):
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=0)
        return self.session.run(self.predict_v, feed_dict={self.states: states})

    def predict(self, states):
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=0)
        outputs = [self.predict_p, self.predict_v]
        return self.session.run(outputs, feed_dict={self.states: states})

    def train_push(self, s, a, r, s2):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s2 is None:
                self.train_queue[3].append(NONE_STATE) # terminal state
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s2)
                self.train_queue[4].append(1.)

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH_SZ:
            time.sleep(0) # yield
            return

        with self.lock_queue:
            # more thread could have passed without lock, we can't yield inside lock
            if len(self.train_queue[0]) < MIN_BATCH_SZ:
                return

            s, a, r, s2, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s, a, r, s2, s_mask = map(np.array, [s, a, r, s2, s_mask])

        if len(s) > MAX_BATCH_SZ:
            print('Optimizer alert! Minimizing batch of %d samples!' % len(s))

        v = self.predict_value(s2)
        r = r + GAMMA_N * v * s_mask # set v to 0 if s2 is terminal state

        self.session.run(self.train_op, feed_dict={self.states: s, self.actions: a, self.returns: r})


class Agent:

    def __init__(self, brain, num_actions):
        self.brain = brain
        self.num_actions = num_actions

        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.memory = [] # used for n-step return
        self.R = 0.

    def select_action(self, s):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            policy = self.brain.predict_policy(s)[0]

            a = np.argmax(policy)
            # a = np.random.choice(self.num_actions, p=policy)

            return a

    def train(self, s, a, r, s2):
        def get_sample(memory, n):
            s0, a0, _, _  = memory[0]
            _,  _,  _, sn = memory[n-1]
            return s0, a0, self.R, sn

        self.memory.append((s, a, r, s2))

        # slowly decrease epsilon based on experience
        self.steps += 1
        self.epsilon = max(MIN_EPSILON, MAX_EPSILON + EPSILON_DECAY * self.steps)

        self.R = (self.R + GAMMA_N * r) / GAMMA

        if s2 is None:
            # handle the edge case - if an episode ends in < N-steps
            if len(self.memory) < N_STEP_RETURN:
                n = N_STEP_RETURN - len(self.memory)
                self.R /= (GAMMA ** n)

            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s2 = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s2)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0.

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s2 = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s2)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


class Environment(threading.Thread):

    def __init__(self, brain, num_actions, render=False, train=True):
        # threading.Thread.__init__(self)
        super(Environment, self).__init__()

        self.render = render
        self.stop_signal = False

        self.env = gym.make(GAME)
        self.agent = Agent(brain, num_actions)

        self.train = train
        if not train:
            self.agent.epsilon = 0.

    def initial_state(self):
        observation = self.env.reset()
        image = process_image(observation)
        state = np.stack([image] * IMAGE_STACK, axis=0)
        return state

    def update_state(self, state, observation):
        image = process_image(observation)
        image = np.expand_dims(image, axis=0)
        next_state = np.append(state[1:], image, axis=0)
        return next_state

    def train_one_episode(self):
        s = self.initial_state()

        total_reward = 0
        while True:
            time.sleep(THREAD_DELAY) # yield

            a = self.agent.select_action(s)
            obs, r, done, info = self.env.step(a)
            s2 = self.update_state(s, obs)
            r /= 20.0 # Used in Game: SpaceInvaders-v0

            # terminal state
            if done:
                s2 = None

            self.agent.train(s, a, r, s2)

            s = s2
            total_reward += r

            if done or self.stop_signal:
                break

        print('Total reward: %s' % total_reward)

    def test_one_episode(self):
        s = self.initial_state()

        total_reward = 0
        while True:
            if self.render:
                self.env.render()

            a = self.agent.select_action(s)
            obs, r, done, info = self.env.step(a)
            s2 = self.update_state(s, obs)
            r /= 20.0 # Used in Game: SpaceInvaders-v0

            s = s2
            total_reward += r

            if done or self.stop_signal:
                break

        print('Total reward: %s' % total_reward)

    def run(self):
        if self.train:
            run_episode = self.train_one_episode
        else:
            run_episode = self.test_one_episode

        while not self.stop_signal:
            run_episode()

    def stop(self):
        self.stop_signal = True


class Optimizer(threading.Thread):

    def __init__(self, brain):
        # threading.Thread.__init__(self)
        super(Optimizer, self).__init__()

        self.brain = brain
        self.stop_signal = False

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True


# ------------------ Entry Point ------------------
if __name__ == '__main__':
    env_tmp = gym.make(GAME)
    NUM_STATE   = (IMAGE_STACK, IMAGE_HEIGHT, IMAGE_WIDTH)
    NUM_ACTIONS = env_tmp.action_space.n
    del env_tmp

    brain = Brain(NUM_STATE, NUM_ACTIONS)

    envs = [Environment(brain, NUM_ACTIONS) for i in range(NUM_ENV_THREADS)]
    opts = [Optimizer(brain)                for i in range(NUM_OPT_THREADS)]

    for opt in opts:
        opt.start()

    for env in envs:
        env.start()

    time.sleep(TRAIN_TIME)

    for env in envs:
        env.stop()
    for env in envs:
        env.join()

    for opt in opts:
        opt.stop()
    for opt in opts:
        opt.join()

    print('\nTraining Finished!\n')

    env_test = Environment(brain, NUM_ACTIONS, render=True, train=False)
    while True:
        env_test.test_one_episode()

        input_str = input('Play again? [Y/N]')
        if input_str in ('n', 'N'):
            break

