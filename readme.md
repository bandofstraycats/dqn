# Deep Q-network algorithm

Deep Q-network algorithm is a reinforcement learning algorithm with Q-values computed by a neural network.
This implementation is based on the original work of [1].
It supports environments that implement Gym API.

## Installation

### Requirements

Python 3.6+, numpy, Tensorflow, Keras, Gym

`pip install numpy tensorflow keras gym[all]`

## Example training Q-network with LunarLander-v2 environment

Q-network is a 2-hidden layer neural network.

`python dqn.py --run-name run-test --env LunarLander-v2 --target --gamma 0.99 --target-update-steps 100 --train-steps 500000 --test-steps 10000 --nb-epoch 1 --log-dir logs --log-steps 10000 --hidden-size 64 --num-hidden 2 --eps-min 0.01 --eps-decay 0.995 --eps-update-steps 1000`

Training and Test evaluation metrics in Tensorboard
`tensorboard --logdir=./logs --port=PORT`

## Example 

## References

[1] Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, Graves A, Riedmiller M, Fidjeland AK, Ostrovski G, Petersen S. Human-level control through deep reinforcement learning. Nature. 2015 Feb;518(7540):529-33.