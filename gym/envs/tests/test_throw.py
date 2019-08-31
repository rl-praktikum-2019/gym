import numpy as np
import gym
import time

episodes = 1000
episode_length = 20000


env = gym.make("ThrowBall-v0")

obs = env.reset()

for j in range(episode_length):
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
