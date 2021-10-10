import pytest

from gym_navigation.envs.navigation import Navigation


def test_invalid_track_id():
    with pytest.raises(ValueError):
        _ = Navigation(-1)
