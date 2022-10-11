import numpy as np
import random
from enum import Enum
import argparse

from collections import deque
from datetime import datetime
import os

from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import gym

import hyperparams, policy

class Mode(Enum):
    def __str__(self):
        return str(self.name)

    train = 1
    predict = 2

class DQNAgent:
    def __init__(self, state_size, action_size, use_policy, hyperparams, use_target_model=True):
        self.state_size = state_size
        self.action_size = action_size

        self.hyperparams = hyperparams

        self.policy = getattr(policy, use_policy)(action_size, hyperparams)

        self.model = self.build_q_network()
        self.use_target_model = use_target_model

        if self.use_target_model:
            self.target_model = self.build_q_network()

        if self.use_target_model:
            self.update_target_model()

        self.memory = deque(maxlen=self.hyperparams.replay_memory_size)

    def build_q_network(self):
        inputs = Input(shape=self.state_size)
        hidden = inputs
        for _ in range(self.hyperparams.num_hidden):
            hidden = Dense(self.hyperparams.hidden_size, activation='relu')(hidden)
        final = Dense(self.action_size, activation='linear')(hidden)
        model = Model(inputs=inputs, outputs=final)
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        expand_state = np.expand_dims(state, 0)
        q_values = self.model.predict(expand_state)

        return self.policy.get_action(q_values)

    def add_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def clear_memory(self):
        self.memory.clear()

    def train_model(self):
        # randomly sample a mini-batch of transition from replay memory
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

        target = self.model.predict(update_input)

        if self.use_target_model:
            target_val = self.target_model.predict(update_target)
        else:
            target_val = self.model.predict(update_target)

        for i in range(self.hyperparams.batch_size):
            # Bellman update
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.hyperparams.gamma * (
                        np.amax(target_val[i]))  # target_val[i][np.argmax(target_val[i])])

        return self.model.train_on_batch(update_input, target)


def write_log(writer, names, values, iteration):
    for name, value in zip(names, values):
        if value is None:
            continue
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, iteration)
        writer.flush()

def train_eval_model(agent, steps, start_step=0, mode=Mode.train):
    losses = []
    q_values = []
    returns = []
    rewards = []
    episode_lens = []
    step = start_step
    while step < start_step + steps:
        done = False
        total_reward = 0
        episode_len = 0
        loss = None
        state = env.reset()
        while not done:
            if args.render:
                env.render()

            action, q_value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            if mode == Mode.train:
                agent.add_transition(state, action, reward, next_state, done)

                loss = agent.train_model()

                if args.policy == 'EpsGreedyPolicy' and step % args.eps_update_steps == 0:
                    agent.policy.update_policy()

                if agent.use_target_model and step % args.target_update_steps == 0:
                    agent.update_target_model()

            state = next_state
            episode_len += 1
            step += 1
            total_reward += reward
            rewards.append(reward)

            if loss:
                losses.append(loss)
            if q_value:
                q_values.append(q_value)

            if done:
                returns.append(total_reward)
                episode_lens.append(episode_len)

            if returns and step % args.log_steps == 0:
                names = ['step', 'avg_return', 'max_return_of_last_100_episodes',
                         'avg_return_of_last_100_episodes', 'avg_episode_len', 'avg_loss', 'avg_q_value', 'epsilon']
                prefix_names = ['/'.join([str(mode), n]) for n in names]

                values = [step,
                          np.average(returns),
                          max(returns[-100:]),
                          np.average(returns[-100:]),
                          np.average(episode_lens),
                          np.average(losses) if losses else None,
                          np.average(q_values) if q_values else None,
                          agent.policy.epsilon]
                write_log(writer, prefix_names, values, step)
                print(list(zip(prefix_names, values)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Q-network algorithm')
    parser.add_argument('--run-name', help='run name', default='', required=True)
    parser.add_argument('--env', help='env name', default='Breakout-ram-v0', required=True)
    parser.add_argument('--train-steps', help='# of training steps', default=1000, type=int)
    parser.add_argument('--test-steps', help='# of evaluation steps', default=1000, type=int)
    parser.add_argument('--render', dest='render', action='store_true', help='Render environment')
    parser.add_argument('--policy', help='EpsGreedyPolicy or SoftmaxPolicy', default='EpsGreedyPolicy')
    # stability
    parser.add_argument('--target', dest='target_dqn', action='store_true', default=True, help='Use target network')
    parser.add_argument('--target-update-steps', help='# of steps to update target network', default=100, type=int)
    # exploration
    parser.add_argument('--eps-update-steps', help='# of steps to update exploration epsilon', default=100, type=int)
    parser.add_argument('--eps-start', help='Start exploration epsilon', default=1.0, type=float)
    parser.add_argument('--eps-decay', help='Decay of exploration epsilon', default=0.995, type=float)
    parser.add_argument('--eps-min', help='Min exploration epsilon', default=0.1, type=float)
    # io
    parser.add_argument('--log-steps', help='# of steps to log on console and Tensorboard', default=100, type=int)
    parser.add_argument('--load-model', dest='load_model', help='Load Q-network from path', default='', type=str)
    parser.add_argument('--log-dir', help='Log directory', default='logs')
    parser.add_argument('--seed', help='Random seed', default=123, type=int)
    # hyperparams
    parser.add_argument('--gamma', help='Gamma discount factor', default=0.99, type=float)
    # sgd
    parser.add_argument('--batch-size', help='Batch size to train Q-network', default=32, type=int)
    parser.add_argument('--train-start', help='Start training after a given number of steps', default=128, type=int)
    parser.add_argument('--memory-size', help='Replay memory size', default=100000, type=int)
    # q network
    parser.add_argument('--hidden-size', help='Q-network hidden layer size', default=32, type=int)
    parser.add_argument('--num-hidden', help='Q-network number of hidden layers', default=2,
                        type=int)

    args = parser.parse_args()
    print(args)

    # Set random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Load environment
    env = gym.make(args.env)
    print('original state_size', env.observation_space.shape)
    env.seed(args.seed)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    reward_threshold = env.spec.reward_threshold
    max_episode_steps = env.spec.max_episode_steps
    print('state_size', state_size, 'action_size', action_size, 'reward_threshold', reward_threshold,
          'max_episode_steps', max_episode_steps)

    # Initialise run
    sess = K.get_session()
    run_name = "{}-dqn-{}-{}".format(args.env, args.run_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = tf.summary.FileWriter("./" + args.log_dir + "/" + run_name, sess.graph)
    print(run_name)

    # Save config
    config_dir = './config'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    with open(config_dir + '/' + run_name + '.conf', 'w') as f:
        f.write('\n'.join(f'{key}={value}' for key, value in vars(args).items()))

    # Initialise DQNAgent
    hyperparams = hyperparams.HyperParams(args)
    agent = DQNAgent(state_size, action_size, args.policy, hyperparams, args.target_dqn)

    # Load Q-network
    if args.load_model:
        print('Load Q-network', args.load_model)
        agent.model = tf.keras.models.load_model(args.load_model)

    # Initialise policy
    agent.policy = getattr(policy, args.policy)(action_size, hyperparams)

    # Train DQNAgent
    train_epoch_steps = args.train_steps
    train_eval_model(agent, steps=train_epoch_steps, start_step=0, mode=Mode.train)

    # Test DQNAgent
    agent.policy.epsilon = 0.0
    train_eval_model(agent, steps=args.test_steps, start_step=0, mode=Mode.predict)

    # Save Q-network
    if args.train_steps > 0:
        models_dir = './models'
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        save_model_path = models_dir + '/' + run_name + ".h5"
        print('Save Q-network to', save_model_path)
        agent.model.save(save_model_path)
