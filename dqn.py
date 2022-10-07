import numpy as np
import sys
import random
from enum import Enum
import argparse

from collections import deque
from datetime import datetime
import math

from skimage.transform import resize
from skimage.color import rgb2gray

from keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import gym
from gym import spaces

import hyperparams, policy

class Mode(Enum):
    def __str__(self):
        return str(self.name)

    train = 1
    valid = 2
    predict = 3

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

    def _observation(self, observation):
        observation = resize(rgb2gray(observation), (84, 84))
        return np.reshape(observation, (self.height, self.width) + (1,))


class WarpGrid(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.height = env.observation_space.shape[0]
        self.width = env.observation_space.shape[1]
        self.flat_shape = self.height * self.width
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.flat_shape, ))

    def _observation(self, observation):
        return np.reshape(observation, (self.flat_shape, ))


# DQN Agent for Atari
class DQNAgent:
    def __init__(self, state_size, action_size, use_policy, hyperparams, use_conv_nets=False,
                 use_target_model=False, use_double_model=False, use_dueling_model=False):
        self.use_target_model = use_target_model
        self.use_dueling_model = use_dueling_model
        self.use_double_model = use_double_model
        self.use_conv_nets = use_conv_nets

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # policy
        self.policy = getattr(policy, use_policy)(action_size, hyperparams)

        # hyperparams
        self.hyperparams = hyperparams
        # create replay memory using deque
        self.memory = deque(maxlen=self.hyperparams.replay_memory_size)

        # create main model
        self.model = self.build_model()

        # create target model
        if self.use_target_model or self.use_double_model:
            self.target_model = self.build_model()

        # update only if target network
        if self.use_target_model:
            self.update_target_model()

    def build_model(self):
        if self.use_dueling_model:
            return self.build_dueling_model()
        else:
            return self.build_dqn_model()

    def build_hidden_layers(self, inputs):
        if self.use_conv_nets:
            net = Conv2D(self.hyperparams.hidden_size, (self.hyperparams.kernel, self.hyperparams.kernel),
                         strides=(self.hyperparams.stride, self.hyperparams.stride), activation='relu',
                         input_shape=self.state_size)(inputs)
            for i in range(self.hyperparams.num_hidden):
                net = Conv2D(self.hyperparams.hidden_size,
                             (self.hyperparams.kernel/math.pow(2, i+1), self.hyperparams.kernel/math.pow(2, i+1)),
                             strides=(self.hyperparams.stride/math.pow(2, i+1), self.hyperparams.stride/math.pow(2, i+1)), activation='relu')(net)
            net = Flatten()(net)
            net = Dense(self.hyperparams.hidden_size, activation='relu')(net)
        else:
            net = Dense(self.hyperparams.hidden_size, input_shape=self.state_size, activation='relu')(inputs)
            for _ in range(self.hyperparams.num_hidden):
                net = Dense(self.hyperparams.hidden_size, activation='relu')(net)
            #net = Dense(self.action_size, activation='linear')(net)
        return net

    def build_dqn_model(self):
        inputs = Input(shape=self.state_size)
        hidden = self.build_hidden_layers(inputs)
        final = Dense(self.action_size, activation='linear')(hidden)
        inputs_list = [inputs]
        model = Model(inputs=inputs_list, outputs=final)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.hyperparams.learning_rate))
        return model

    def build_dueling_model(self):
        inputs = Input(shape=self.state_size)
        hidden = self.build_hidden_layers(inputs)
        y = Dense(self.action_size + 1, activation='linear')(hidden)
        final = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                       output_shape=(self.action_size,))(y)
        model = Model(inputs=inputs, outputs=final)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.hyperparams.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        expand_state = np.expand_dims(state, 0)
        q_values = self.model.predict(expand_state)

        return self.policy.get_action(q_values)

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_memory(self):
        self.memory.clear()

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.hyperparams.train_start:
            return None
        batch_size = min(self.hyperparams.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((self.hyperparams.batch_size,) + self.state_size)
        update_target = np.zeros((self.hyperparams.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        for i in range(self.hyperparams.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(
            update_input)  # just to get all the same values except for the action taken to get zero update on them

        if self.use_target_model:
            target_val = self.target_model.predict(update_target)
        else:
            target_val = self.model.predict(update_target)

        if self.use_double_model:
            double_target_val = self.target_model.predict(update_target)

        for i in range(self.hyperparams.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # need to come first to access to the current q values
                if self.use_double_model:
                    target[i][action[i]] = reward[i] + self.hyperparams.gamma * (
                        double_target_val[i][np.argmax(target_val[i])])
                else:
                    target[i][action[i]] = reward[i] + self.hyperparams.gamma * (
                        np.amax(target_val[i]))  # target_val[i][np.argmax(target_val[i])])

        # and do the model fit!
        return self.model.train_on_batch(update_input, target)


def write_log(writer, names, values, iteration):
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, iteration)
        writer.flush()

def train_eval_model(agent, steps, start_step = 0, mode=Mode.train):
    losses = []
    q_values = []
    returns = []
    rewards = []
    step = start_step
    while step < start_step + steps:
        done = False
        total_reward = 0
        episode_len = 0
        state = env.reset()
        while not done:
            if args.render:
                env.render()

            # get action for the current state and go one step in environment
            action, q_value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # every time step do the training
            if mode == Mode.train:
                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)

                loss = agent.train_model()

                if args.policy == 'EpsGreedyPolicy' and step % args.eps_update_steps == 0:
                    agent.policy.update_policy()

                if step % args.target_update_steps == 0:
                    if agent.use_target_model or agent.use_double_model:
                        agent.update_target_model()

                if step % args.value_target_update_steps == 0:
                    agent.update_target_value_model()
            else:
                loss = 1

            state = next_state
            episode_len += 1
            step += 1
            total_reward += reward
            rewards.append(reward)

            if loss:
                losses.append(loss)
            if q_value:
                q_values.append(q_value)

        returns.append(total_reward)

        if losses and returns:
            names = ['step', 'avg_return', 'max_return_of_last_100_episodes',
                     'avg_return_of_last_100_episodes', 'avg_loss', 'avg_q_value', 'episode_len', 'epsilon']
            prefix_names = ['/'.join([str(mode), n]) for n in names]

            values = [step,
                      np.average(returns),
                      max(returns[-100:]),
                      np.average(returns[-100:]),
                      np.average(losses),
                      np.average(q_values),
                      episode_len,
                      agent.policy.epsilon]
            write_log(writer, prefix_names, values, step)
            print(list(zip(prefix_names, values)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN for playing Atari')
    parser.add_argument('--run-name', help='run name', default='', required=True)
    parser.add_argument('--env', help='env name', default='Breakout-ram-v0', required=True)
    parser.add_argument('--max-steps', help='# of episodes to play', default=30000, type=int)
    parser.add_argument('--eval-steps', help='# of episodes to play for final evaluation', default=10000, type=int)
    parser.add_argument('--train-epoch-steps', help='# of episodes to play for final evaluation', default=10000, type=int)
    parser.add_argument('--eval-epoch-steps', help='# of episodes to play for final evaluation', default=5000, type=int)
    parser.add_argument('--render', dest='render', action='store_true', help='Render episodes')
    parser.add_argument('--policy', help='EpsGreedyPolicy or SoftmaxPolicy', default='EpsGreedyPolicy')
    # stability
    parser.add_argument('--target', dest='target_dqn', action='store_true', help='Target DQN')
    parser.add_argument('--target-update-steps', help='# of steps to update target network', default=100, type=int)
    parser.add_argument('--double', dest='double_dqn', action='store_true', help='Double DQN')
    parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Dueling DQN')
    # exploration
    parser.add_argument('--eps-update-steps', help='# of steps to update exploration epsilon', default=100, type=int)
    parser.add_argument('--eps-start', help='Start exploration epsilon', default=1.0, type=float)
    parser.add_argument('--eps-decay', help='Decay of exploration epsilon', default=0.9996, type=float)
    parser.add_argument('--eps-min', help='Min exploration epsilon', default=0.1, type=float)
    # io
    parser.add_argument('--save-freq', help='# of episodes to save model', default=1000, type=int)
    parser.add_argument('--log-steps', help='# of steps to log on console', default=100, type=int)
    parser.add_argument('--load-model', dest='load_model', help='load previous model', default='', type=str)
    parser.add_argument('--log-dir', help='Log directory', default='logs')
    parser.add_argument('--seed', help='Random seed', default=123, type=int)
    # hyperparams
    parser.add_argument('--gamma', help='Gamma discount factor for LT reward', default=0.99, type=float)
    # sgd
    parser.add_argument('--lr', help='Learning for Q function', default=0.001, type=float)
    parser.add_argument('--batch-size', help='Batch size for learning for Q function', default=32, type=int)
    parser.add_argument('--train-start', help='Start training after this number of iterations', default=128, type=int)
    parser.add_argument('--memory-size', help='Replay memory size', default=100000, type=int)
    # q network
    parser.add_argument('--use-conv-nets', dest='use_conv_nets', action='store_true', help='Use ConvNet for DQN')
    parser.add_argument('--hidden-size', help='DNN hidden layer size', default=512, type=int)
    parser.add_argument('--num-hidden', help='DNN number of hidden layers excluding input and output', default=2,
                        type=int)
    parser.add_argument('--kernel', help='ConvNet kernel size', default=8, type=int)
    parser.add_argument('--stride', help='ConvNet stride size', default=4, type=int)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = gym.make(args.env)
    print('original state_size', env.observation_space.shape)
    if args.env.startswith('grid'):
        env = WarpGrid(env)
    elif args.use_conv_nets:
        env = WarpFrame(env)
    env.seed(args.seed)
    # env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)
    # get size of state and action from environment
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    reward_threshold = env.spec.reward_threshold
    max_episode_steps = env.spec.max_episode_steps
    print('state_size', state_size, 'action_size', action_size, 'reward_threshold', reward_threshold,
          'max_episode_steps', max_episode_steps)

    hyperparams = hyperparams.HyperParams(args)
    agent = DQNAgent(state_size, action_size, args.policy, hyperparams, args.use_conv_nets,
                     args.target_dqn, args.double_dqn, args.dueling_dqn)

    sess = K.get_session()
    run_name = "{}-dqn-{}-{}".format(args.env, args.run_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = tf.summary.FileWriter("./" + args.log_dir + "/" + run_name, sess.graph)
    print(run_name)

    # Save config
    with open('./config/' + run_name + '.conf', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # Load agent weights
    if args.load_model:
        print('Load agent model weights', args.load_model)
        agent.model.load_weights(args.load_model)

    # Train DQN
    agent.policy = getattr(policy, args.policy)(action_size, hyperparams)

    train_epoch_steps = args.train_epoch_steps
    eval_epoch_steps = args.eval_epoch_steps
    nb_epoch = args.max_steps // train_epoch_steps if train_epoch_steps > 0 else 0
    for epoch in range(nb_epoch):
        print('Epoch', epoch)
        train_eval_model(agent, steps=train_epoch_steps, start_step=epoch * train_epoch_steps, mode=Mode.train)
        save_epsilon = agent.policy.epsilon
        if epoch < nb_epoch-1:
            # Skip last eval
            agent.policy.epsilon = 0.0
            train_eval_model(agent, steps=eval_epoch_steps, start_step=epoch * eval_epoch_steps, mode=Mode.valid)
            agent.policy.epsilon = save_epsilon

    # final
    agent.policy.epsilon = 0.0
    train_eval_model(agent, steps=args.eval_steps, start_step=0, mode=Mode.predict)

    if args.max_steps > 0:
        save_model_path = "./models/" + run_name + ".h5"
        print('Save value model to', save_model_path)
        agent.model.save_weights(save_model_path)