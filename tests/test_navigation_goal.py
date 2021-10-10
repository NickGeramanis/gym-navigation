import pytest

from gym_navigation.envs.navigation_goal import NavigationGoal


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationGoal(-1)
