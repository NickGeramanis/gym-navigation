import math

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_navigation.enums.action import Action
from gym_navigation.envs.navigation_goal import NavigationGoal

ATOL = 0.06


def test_do_check_if_terminated_true_collision():
    env = NavigationGoal()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])
    env._distance_from_goal = 10

    done = env._do_check_if_terminated()

    assert done


def test_do_check_if_terminated_true_goal_reached():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 0.01

    done = env._do_check_if_terminated()

    assert done


def test_do_check_if_terminated_false():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 10

    done = env._do_check_if_terminated()

    assert not done


def test_do_calculate_reward_if_collision():
    env = NavigationGoal()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])
    env._distance_from_goal = 10

    reward = env._do_calculate_reward(Action.FORWARD.value)

    assert reward == -200


def test_do_calculate_reward_if_goal_reached():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 0.01

    reward = env._do_calculate_reward(Action.FORWARD.value)

    assert reward == 200


def test_do_calculate_reward_if_action():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._previous_distance_from_goal = 1.1
    env._distance_from_goal = 1

    reward = env._do_calculate_reward(Action.FORWARD.value)

    assert math.isclose(reward, 1)


def test_do_init_environment():
    env = NavigationGoal()
    env._world = env._track.walls

    env._do_init_environment()

    assert env._pose is not None and env._goal is not None


def test_sanity():
    env = gym.make('gym_navigation:NavigationGoal-v0', track_id=2)
    check_env(env.unwrapped)
