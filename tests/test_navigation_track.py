import math

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_navigation.enums.action import Action
from gym_navigation.envs.navigation_track import NavigationTrack
from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose

ATOL = 0.06


def test_do_perform_action_forward():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)
    env._world = env._track.walls

    env._do_perform_action(Action.FORWARD.value)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.7, abs_tol=ATOL)
    assert math.isclose(env._pose.yaw, 0, abs_tol=ATOL)


def test_do_perform_action_rotate_left():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)
    env._world = env._track.walls

    env._do_perform_action(Action.ROTATE_LEFT.value)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.54, abs_tol=ATOL)
    assert math.isclose(env._pose.yaw, -0.2, abs_tol=ATOL)


def test_do_perform_action_rotate_right():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)
    env._world = env._track.walls

    env._do_perform_action(Action.ROTATE_RIGHT.value)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.54, abs_tol=ATOL)
    assert math.isclose(env._pose.yaw, 0.2, abs_tol=ATOL)


def test_do_check_if_terminated_true():
    env = NavigationTrack()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])

    done = env._do_check_if_terminated()

    assert done


def test_do_check_if_terminated_false():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    done = env._do_check_if_terminated()

    assert not done


def test_do_calculate_reward_if_done():
    env = NavigationTrack()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])

    reward = env._do_calculate_reward(Action.FORWARD.value)

    assert reward == -200


def test_do_calculate_reward_if_forward():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    reward = env._do_calculate_reward(Action.FORWARD.value)

    assert reward == 5


def test_do_calculate_reward_if_rotate():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    reward1 = env._do_calculate_reward(Action.ROTATE_RIGHT.value)
    reward2 = env._do_calculate_reward(Action.ROTATE_LEFT.value)

    assert reward1 == reward2 == -0.5


def test_do_init_environment():
    env = NavigationTrack()
    env._world = env._track.walls

    env._do_init_environment()

    assert env._pose is not None


def test_sanity():
    env = gym.make('gym_navigation:NavigationTrack-v0', track_id=1)
    check_env(env.unwrapped)
