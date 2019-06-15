# How to create new environments for Gym

* Create a new repo called gym-foo, which should also be a PIP package.

* A good example is https://github.com/openai/gym-soccer.

* It should have at least the following files:
  ```sh
  gym-foo/
    README.md
    setup.py
    gym_foo/
      __init__.py
      envs/
        __init__.py
        foo_env.py
        foo_extrahard_env.py
  ```

* `gym-foo/setup.py` should have:

  ```python
  from setuptools import setup

  setup(name='gym_foo',
        version='0.0.1',
        install_requires=['gym']  # And any other dependencies foo needs
  )
  ```

* `gym-foo/gym_foo/__init__.py` should have:
  ```python
  from gym.envs.registration import register

  register(
      id='foo-v0',
      entry_point='gym_foo.envs:FooEnv',
  )
  register(
      id='foo-extrahard-v0',
      entry_point='gym_foo.envs:FooExtraHardEnv',
  )
  ```

* `gym-foo/gym_foo/envs/__init__.py` should have:
  ```python
  from gym_foo.envs.foo_env import FooEnv
  from gym_foo.envs.foo_extrahard_env import FooExtraHardEnv
  ```

* `gym-foo/gym_foo/envs/foo_env.py` should look something like:
  ```python
  import gym
  from gym import error, spaces, utils
  from gym.utils import seeding

  class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
      ...
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human'):
      ...
    def close(self):
      ...
  ```

* After you have installed your package with `pip install -e gym-foo`, you can create an instance of the environment with `gym.make('gym_foo:foo-v0')`


# How to create new robotics environment using Mujoco for Gym

Structure:
* gym/gym/envs/robotics: Holds envs for robotics. Each env is inherits on `class MyEnv(robot_env.RobotEnv)`. The env performs actions.

* gym/gym/envs/robotics/assets: Holds textures, models and physics object definitions (joints, actuators etc). Work in this directory for new models and visual env setup. Watch out for distinct ids and names in xml-tags.

* gym/gym/envs/robotics/hand: Holds the task logic f.e. for manipulating a object, reaching a position, or interactions between multiple hands. Work in this directory for new environment logics. (compute_reward, success, goal_distance)

TODO in MD:
- xml aufbau kapieren

PS: Für uns sind alle xml und .py mit MultiHand im Namen gedacht.


