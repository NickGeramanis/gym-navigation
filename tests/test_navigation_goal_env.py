import pytest

from gym_navigation.envs.navigation_goal_env import NavigationGoalEnv


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationGoalEnv(-1)
