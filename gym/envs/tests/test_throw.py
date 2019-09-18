import numpy as np
import gym
import time

episodes = 1000
episode_length = 20000
pr
env = gym.make("ThrowBall-v0", reward_type='dense')

for i in range(episodes):
    obs = env.reset()

    for j in range(episode_length):
        start = time.time()

        obs, reward, done, info = env.step(env.action_space.sample())
        print(env.env.sim.data.get_body_xpos('trajectory:lower'))
        print(env.env.sim.data.get_body_xpos('trajectory:upper'))

        env.render()

        end = time.time()
        print("Time:", end - start)

        if done:  # Ball dropped
            break

env.close()
