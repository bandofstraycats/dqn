import numpy as np

class Policy(object):
    def __init__(self, action_size, hyperparams):
        self.hyperparams = hyperparams
        self.action_size = action_size

    def get_action(self, **kwargs):
        raise NotImplementedError()

    def update_policy(self, **kwargs):
        raise NotImplementedError()


class EpsGreedyPolicy(Policy):

    def __init__(self, action_size, hyperparams):
        super(EpsGreedyPolicy, self).__init__(action_size, hyperparams)
        self.epsilon = self.hyperparams.epsilon_start

    def get_action(self, q_values):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size), None
        else:
            return np.argmax(q_values[0]), np.max(q_values[0])

    def update_policy(self):
        if self.epsilon > self.hyperparams.epsilon_min:
            self.epsilon *= self.hyperparams.epsilon_decay