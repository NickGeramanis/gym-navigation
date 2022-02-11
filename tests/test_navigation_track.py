import math

import numpy as np
import pytest

from gym_navigation.envs.navigation_track import NavigationTrack
from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationTrack(-1)


def test_forward_action():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)

    env._do_perform_action(env._FORWARD)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.7, abs_tol=0.06)
    assert math.isclose(env._pose.yaw, 0, abs_tol=0.06)


def test_yaw_left_action():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)

    env._do_perform_action(env._YAW_LEFT)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.54, abs_tol=0.06)
    assert math.isclose(env._pose.yaw, -0.2, abs_tol=0.06)


def test_yaw_right_action():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0)

    env._do_perform_action(env._YAW_RIGHT)

    assert math.isclose(env._pose.position.x_coordinate, 0)
    assert math.isclose(env._pose.position.y_coordinate, 8.54, abs_tol=0.06)
    assert math.isclose(env._pose.yaw, 0.2, abs_tol=0.06)


def test_do_update_observation():
    env = NavigationTrack()
    env._pose = Pose(Point(0, 8.5), 0.78539816339)

    env._do_update_observation()

    assert np.allclose(env._observation,
                       [2.12132034356, 1.5, 2.12132034356, 10, 2.12132034356],
                       atol=0.06)


def test_do_check_if_done_true():
    env = NavigationTrack()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])

    done = env._do_check_if_done()

    assert done


def test_do_check_if_done_false():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    done = env._do_check_if_done()

    assert not done


def test_do_calculate_reward_if_done():
    env = NavigationTrack()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])

    reward = env._do_calculate_reward(env._FORWARD)

    assert reward == -200


def test_do_calculate_reward_if_forward():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    reward = env._do_calculate_reward(env._FORWARD)

    assert reward == 5


def test_do_calculate_reward_if_yaw():
    env = NavigationTrack()
    env._ranges = np.array([2, 1, 10, 4, 5])

    reward1 = env._do_calculate_reward(env._YAW_RIGHT)
    reward2 = env._do_calculate_reward(env._YAW_LEFT)

    assert reward1 == reward2 == -0.5
