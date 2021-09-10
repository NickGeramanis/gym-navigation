import pytest

from gym_navigation.envs.navigation_goal_env import NavigationGoalEnv


class TestNavigationGoalEnv:
    def test_invalid_track_id(self):
        with pytest.raises(ValueError):
            _ = NavigationGoalEnv(-1)

    def test_valid_track_id(self):
        _ = NavigationGoalEnv(1)
