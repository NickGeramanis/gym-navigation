import pytest

from gym_navigation.envs.navigation_track import NavigationTrack


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = NavigationTrack(-1)
