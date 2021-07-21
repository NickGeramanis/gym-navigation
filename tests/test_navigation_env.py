import math
import random

import numpy as np
import pytest
from gym_navigation.envs.navigation_env import NavigationEnv


class TestNavigationEnv:
    TOLERANCE = 0.000000000001

    def test_invalid_track_id(self):
        with pytest.raises(Exception):
            env = NavigationEnv(-1)

    def test_get_point(self):
        # movement in the 1st quadrant
        env = NavigationEnv(1)
        x0 = -1
        y0 = 1
        angle = -math.pi / 4
        d = 5

        x1, y1 = env.get_point(x0, y0, angle, d)

        x1_correct = x0 - d * math.sin(abs(angle))
        y1_correct = y0 + d * math.cos(abs(angle))

        assert math.isclose(x1, x1_correct, abs_tol=self.TOLERANCE)
        assert math.isclose(y1, y1_correct, abs_tol=self.TOLERANCE)

    def test_perform_action_forward(self):
        # 99.7% success rate due to random noise (tolerance is 3SD)
        env = NavigationEnv(1)
        env.pose[0] = random.uniform(-10, 10)
        env.pose[1] = random.uniform(-10, 10)
        env.pose[2] = random.uniform(-math.pi / 2, math.pi / 2)
        startin_pose = np.copy(env.pose)

        env.perform_action(env.FORWARD)

        distance = math.dist(env.pose[:2], startin_pose[:2])
        assert math.isclose(distance, env.FORWARD_LINEAR_SHIFT,
                            abs_tol=3 * env.SHIFT_STANDARD_DEVIATION)
        assert math.isclose(env.pose[2], startin_pose[2],
                            abs_tol=3 * env.SHIFT_STANDARD_DEVIATION)

    def test_perform_action_yaw_left(self):
        # 99.7% success rate due to random noise (tolerance is 3SD)
        env = NavigationEnv(1)
        env.pose[0] = random.uniform(-10, 10)
        env.pose[1] = random.uniform(-10, 10)
        env.pose[2] = random.uniform(-math.pi / 2, math.pi / 2)
        startin_pose = np.copy(env.pose)

        env.perform_action(env.YAW_LEFT)

        distance = math.dist(env.pose[:2], startin_pose[:2])
        assert math.isclose(distance, env.YAW_LINEAR_SHIFT,
                            abs_tol=3 * env.SHIFT_STANDARD_DEVIATION)
        assert math.isclose(
            env.pose[2], startin_pose[2],
            abs_tol=env.YAW_ANGULAR_SHIFT + 3 * env.SHIFT_STANDARD_DEVIATION)

    def test_perform_action_yaw_right(self):
        # 99.7% success rate due to random noise (tolerance is 3SD)
        env = NavigationEnv(1)
        env.pose[0] = random.uniform(-10, 10)
        env.pose[1] = random.uniform(-10, 10)
        env.pose[2] = random.uniform(-math.pi / 2, math.pi / 2)
        startin_pose = np.copy(env.pose)

        env.perform_action(env.YAW_RIGHT)

        distance = math.dist(env.pose[:2], startin_pose[:2])
        assert math.isclose(distance, env.YAW_LINEAR_SHIFT,
                            abs_tol=3 * env.SHIFT_STANDARD_DEVIATION)
        assert math.isclose(
            env.pose[2], startin_pose[2],
            abs_tol=env.YAW_ANGULAR_SHIFT + 3 * env.SHIFT_STANDARD_DEVIATION)

    def test_update_scan(self):
        # 99.7% success rate due to random noise (tolerance is 3SD)
        env = NavigationEnv(1)
        env.reset()
        env.pose[0] = 0
        env.pose[1] = 9
        env.pose[2] = 0

        env.update_scan()

        correct_ranges = [
            10, 1 / math.cos(math.pi / 4), 1, 1 / math.cos(math.pi / 4), 10]
        for i in range(env.N_MEASUREMENTS):
            assert math.isclose(env.ranges[i], correct_ranges[i],
                                abs_tol=3 * env.SENSOR_STANDARD_DEVIATION)

    def test_collision_occured(self):
        env = NavigationEnv(1)
        env.reset()
        env.pose[0] = env.track[0][0][0]
        env.pose[1] = env.track[0][1][0]

        env.update_scan()

        assert env.collision_occurred()
