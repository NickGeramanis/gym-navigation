import numpy as np
import pytest

from gym_navigation.envs.navigation_goal import NavigationGoal
from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationGoal(-1)


def test_do_update_observation():
    env = NavigationGoal()
    env._pose = Pose(Point(0, 8.5), 0.78539816339)
    env._goal = Point(1.5, 8.5)

    env._do_update_observation()

    expected_observation = [2.12132034356,
                            1.5,
                            2.12132034356,
                            10,
                            14.1421356237,
                            1.5,
                            0.78539816339]
    assert np.allclose(env._observation, expected_observation, atol=0.06)


def test_do_check_if_done_true_collision():
    env = NavigationGoal()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])
    env._distance_from_goal = 10

    done = env._do_check_if_done()

    assert done


def test_do_check_if_done_true_goal_reached():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 0.01

    done = env._do_check_if_done()

    assert done


def test_do_check_if_done_false():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 10

    done = env._do_check_if_done()

    assert not done


def test_do_calculate_reward_if_collision():
    env = NavigationGoal()
    env._ranges = np.array([0.5, 1, 0.01, 4, 5])
    env._distance_from_goal = 10

    reward = env._do_calculate_reward(env._FORWARD)

    assert reward == -200


def test_do_calculate_reward_if_goal_reached():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 0.01

    reward = env._do_calculate_reward(env._FORWARD)

    assert reward == 200


def test_do_calculate_reward_if_action():
    env = NavigationGoal()
    env._ranges = np.array([2, 1, 10, 4, 5])
    env._distance_from_goal = 1
    env._observation = np.array([2, 1, 10, 4, 5, 3, 0.5])

    reward = env._do_calculate_reward(env._FORWARD)

    assert reward == 20
