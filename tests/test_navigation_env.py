import pytest

from gym_navigation.envs.navigation_env import NavigationEnv


class TestNavigationEnv:
    def test_invalid_track_id(self):
        with pytest.raises(ValueError):
            _ = NavigationEnv(-1)
