# Used for checking the reward range of specific Atari Game

from __future__ import print_function

import gym


GAME = 'Breakout-v0'
RUN_EPISODES = 20000


if __name__ == '__main__':
    env = gym.make(GAME)

    env.reset()

    rewards = set()
    min_reward = float('+inf')
    max_reward = float('-inf')

    for i in range(RUN_EPISODES):
        s, r, d, _ = env.step(env.action_space.sample())

        rewards.add(r)

        if r < min_reward:
            min_reward = r
        if r > max_reward:
            max_reward = r

        if d:
            env.reset()

    print('\nMin reward: %s, Max reward: %s' % (min_reward, max_reward))
    print('\nRewards:', rewards, '\n')

