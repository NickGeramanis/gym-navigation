import math

import pytest
from gym_navigation.envs.navigation_goal_env import NavigationGoalEnv


class TestNavigationGoalEnv:
    TOLERANCE = 0.0000000001

    def test_get_point(self):
        # movement in the 1st quadrant
        env = NavigationGoalEnv()
        x0 = -1
        y0 = 1
        angle = -math.pi / 4
        d = 5

        x1, y1 = env.get_point(x0, y0, angle, d)

        x1_correct = x0 - d * math.sin(abs(angle))
        y1_correct = y0 + d * math.cos(abs(angle))

        assert math.isclose(x1, x1_correct, abs_tol=self.TOLERANCE)
        assert math.isclose(y1, y1_correct, abs_tol=self.TOLERANCE)
