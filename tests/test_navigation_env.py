import pytest

from gym_navigation.envs.navigation_env import NavigationEnv


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationEnv(-1)
