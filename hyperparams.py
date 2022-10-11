class HyperParams:
    def __init__(self, args):
        # hyper-parameters of DQN
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.train_start = args.train_start
        # replay memory size
        self.replay_memory_size = args.memory_size
        # q-network
        self.hidden_size = args.hidden_size
        self.num_hidden = args.num_hidden
        # policy
        # epsilon greedy
        self.epsilon_start = args.eps_start
        self.epsilon_decay = args.eps_decay
        self.epsilon_min = args.eps_min