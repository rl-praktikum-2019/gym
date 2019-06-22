import os
import numpy as np

from gym import utils
from gym.envs.robotics import hand_env, multi_hand_env
from gym.envs.robotics.utils import robot_get_obs


LEFT_FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]

RIGHT_FINGERTIP_SITE_NAMES = [
    'robot1:S_fftip',
    'robot1:S_mftip',
    'robot1:S_rftip',
    'robot1:S_lftip',
    'robot1:S_thtip',
]
# TODO:_are these still correct after transforming hands
DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,

    'robot1:WRJ1': -0.16514339750464327,
    'robot1:WRJ0': -0.31973286565062153,
    'robot1:FFJ3': 0.14340512546557435,
    'robot1:FFJ2': 0.32028208333591573,
    'robot1:FFJ1': 0.7126053607727917,
    'robot1:FFJ0': 0.6705281001412586,
    'robot1:MFJ3': 0.000246444303701037,
    'robot1:MFJ2': 0.3152655251085491,
    'robot1:MFJ1': 0.7659800313729842,
    'robot1:MFJ0': 0.7323156897425923,
    'robot1:RFJ3': 0.00038520700007378114,
    'robot1:RFJ2': 0.36743546201985233,
    'robot1:RFJ1': 0.7119514095008576,
    'robot1:RFJ0': 0.6699446327514138,
    'robot1:LFJ4': 0.0525442258033891,
    'robot1:LFJ3': -0.13615534724474673,
    'robot1:LFJ2': 0.39872030433433003,
    'robot1:LFJ1': 0.7415570009679252,
    'robot1:LFJ0': 0.704096378652974,
    'robot1:THJ4': 0.003673823825070126,
    'robot1:THJ3': 0.5506291436028695,
    'robot1:THJ2': -0.014515151997119306,
    'robot1:THJ1': -0.0015229223564485414,
    'robot1:THJ0': -0.7894883021600622,
}

# Ensure we get the path separator correct on windows
MULTI_HAND_XML = os.path.join('hand', 'multi_hand_reach.xml')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MultiHandReachEnv(multi_hand_env.MultiHandEnv, utils.EzPickle):
    def __init__(
        self, distance_threshold=0.01, n_substeps=40, relative_control=False,
        initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='sparse',
    ):
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        multi_hand_env.MultiHandEnv.__init__(
            self, MULTI_HAND_XML, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)
        utils.EzPickle.__init__(self)

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for sublist in [LEFT_FINGERTIP_SITE_NAMES, RIGHT_FINGERTIP_SITE_NAMES] for name in sublist]
        #goal = [self.sim.data.get_site_xpos(name) for name in LEFT_FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.left_palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()
        self.right_palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot1:palm')].copy()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        left_thumb_name = 'robot0:S_thtip'
        right_thumb_name = 'robot1:S_thtip'

        finger_names_all = [name for sublist in [LEFT_FINGERTIP_SITE_NAMES, RIGHT_FINGERTIP_SITE_NAMES] for name in sublist if (name != left_thumb_name) & (name != right_thumb_name)]

        left_finger_names = [name for name in LEFT_FINGERTIP_SITE_NAMES if name != left_thumb_name]
        right_finger_names = [name for name in RIGHT_FINGERTIP_SITE_NAMES if name != right_thumb_name]

        left_finger_name = self.np_random.choice(left_finger_names)
        right_finger_name = self.np_random.choice(right_finger_names)

        left_thumb_idx = LEFT_FINGERTIP_SITE_NAMES.index(left_thumb_name)
        left_finger_idx = LEFT_FINGERTIP_SITE_NAMES.index(left_finger_name)

        right_thumb_idx = RIGHT_FINGERTIP_SITE_NAMES.index(right_thumb_name)
        right_finger_idx = RIGHT_FINGERTIP_SITE_NAMES.index(right_finger_name)

        assert (left_thumb_idx != left_finger_idx) & (right_thumb_idx != right_finger_idx)

        # Pick a meeting point above the hand.
        left_meeting_pos = self.left_palm_xpos + np.array([0.0, -0.09, 0.05])
        left_meeting_pos += self.np_random.normal(scale=0.005, size=left_meeting_pos.shape)

        right_meeting_pos = self.right_palm_xpos + np.array([0.0, -0.09, 0.05])
        right_meeting_pos += self.np_random.normal(scale=0.005, size=right_meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [left_thumb_idx, left_finger_idx]:
            offset_direction = (left_meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = left_meeting_pos - 0.005 * offset_direction

        for idx in [right_thumb_idx, right_finger_idx]:
            offset_direction = (right_meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = left_meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal.reshape(10, 3)

        for finger_idx in range(5):
            site_name = 'left_target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(10, 3)
        for finger_idx in range(5):
            site_name = 'left_finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]


        for finger_idx in range(5):
            site_name = 'right_target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(10, 3)
        for finger_idx in range(5):
            site_name = 'right_finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]

        self.sim.forward()
