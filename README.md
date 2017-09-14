# AI for Atari

Deep Reinforcement Learning Algorithms for solving Atari 2600 Games

## Reference
The implementation of Double DQN with Prioritized Experience Replay (Proportional Prioritization) is based on:
- Mnih et al. [Human-level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) [2015.02]
- van Hasselt et al. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) [2015.12]
- Schaul et al. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) [2016.02]

The implementation of Asynchronous Advantage Actor-Critic (A3C) algorithm is based on:
- Mnih et al. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) [2016.06]
- Babaeizadeh et al. [Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU](https://arxiv.org/pdf/1611.06256.pdf) [2017.03]


## Environment
- <b>Python</b> 2.7.x or 3.6.x
- <b>NumPy</b> 1.13.1
- <b>TensorFlow</b> 1.0.* or 1.1.* or 1.2.* or 1.3.*
- <b>Keras</b> 2.0.8
- <b>SciPy</b> 0.19.1 (For image pre-processing)
- <b>H5py</b> 2.7.1 (For saving or loading Keras model)
- <b>Gym</b> 0.9.3 (Provides Atari 2600 Games)
