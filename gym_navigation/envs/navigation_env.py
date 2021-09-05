import math
import random
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class NavigationEnv(gym.Env):
    __N_ACTIONS = 3
    __FORWARD = 0
    __YAW_RIGHT = 1
    __YAW_LEFT = 2

    __FORWARD_LINEAR_SHIFT = 0.2  # m
    __YAW_LINEAR_SHIFT = 0.04  # m
    __YAW_ANGULAR_SHIFT = 0.2  # rad

    __SHIFT_STANDARD_DEVIATION = 0.02
    __SENSOR_STANDARD_DEVIATION = 0.01

    __WALL_DISTANCE_THRESHOLD = 0.4

    __COLLISION_REWARD = -200
    __FORWARD_REWARD = +5
    __YAW_REWARD = -0.5

    __SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)

    __SCAN_RANGE_MAX = 30.0
    __SCAN_RANGE_MIN = 0.2
    __N_MEASUREMENTS = len(__SCAN_ANGLES)

    __TRACK1 = (
        ((-10, -10), (-10, 10)),
        ((-10, 10), (10, 10)),
        ((10, 10), (10, -1.5)),
        ((10, 1.5), (-1.5, -1.5)),
        ((1.5, 1.5), (-1.5, -10)),
        ((1.5, -10), (-10, -10)),

        ((-7, -7), (-7, 7)),
        ((-7, 7), (7, 7)),
        ((7, 7), (7, 1.5)),
        ((7, -1.5), (1.5, 1.5)),
        ((-1.5, -1.5), (1.5, -7)),
        ((-1.5, -7), (-7, -7))
    )

    __SPAWNABLE_AREA1 = (
        ((-8.5, -8.5), (-8.5, 8.5)),
        ((-8.5, 8.5), (8.5, 8.5)),
        ((8.5, 8.5), (0, 8.5)),
        ((0, 8.5), (0, 0)),
        ((0, 0), (-8.5, 0)),
        ((-8.5, 0), (-8.5, -8.5))
    )

    def __init__(self, track_id: int = 1) -> None:
        if track_id == 1:
            self.__track = self.__TRACK1
            self.__spawnable_area = self.__SPAWNABLE_AREA1
        else:
            raise ValueError('Invalid track id')

        self.__ranges = np.empty((self.__N_MEASUREMENTS,))

        # Pose = (x, y, yaw)
        # Note that yaw is measured from the y axis and E [-pi, pi].
        self.__pose = np.empty((3,))

        self.__scan_lines = np.empty((self.__N_MEASUREMENTS,), dtype=object)
        self.__scan_points = np.empty((self.__N_MEASUREMENTS,), dtype=object)

        self.action_space = spaces.Discrete(self.__N_ACTIONS)

        self.observation_space = spaces.Box(
            low=self.__SCAN_RANGE_MAX, high=self.__SCAN_RANGE_MIN,
            shape=(self.__N_MEASUREMENTS,), dtype=np.float32)

    @staticmethod
    def get_point(x0: float, y0: float,
                  angle: float, d: float) -> Tuple[float, float]:
        if angle == 0:
            y1 = y0 + d
            x1 = x0
        elif abs(angle) == math.pi:
            y1 = y0 - d
            x1 = x0
        else:
            m = math.tan(math.pi / 2 - angle)
            if angle < 0:
                x1 = x0 - math.sqrt(d ** 2 / (m ** 2 + 1))
            else:
                x1 = x0 + math.sqrt(d ** 2 / (m ** 2 + 1))
            y1 = y0 - m * (x0 - x1)

        return x1, y1

    def __perform_action(self, action: int) -> None:
        linear_shift_noise = random.gauss(0, self.__SHIFT_STANDARD_DEVIATION)
        angular_shift_noise = random.gauss(0, self.__SHIFT_STANDARD_DEVIATION)

        if action == self.__FORWARD:
            d = self.__FORWARD_LINEAR_SHIFT + linear_shift_noise
            self.__pose[2] += angular_shift_noise
        elif action == self.__YAW_RIGHT:
            d = self.__YAW_LINEAR_SHIFT + linear_shift_noise
            self.__pose[2] += self.__YAW_ANGULAR_SHIFT + angular_shift_noise
        elif action == self.__YAW_LEFT:
            d = self.__YAW_LINEAR_SHIFT + linear_shift_noise
            self.__pose[2] -= self.__YAW_ANGULAR_SHIFT + angular_shift_noise
        else:
            raise ValueError(f'{action} ({type(action)}) invalid')

        # Yaw must E [-pi,pi].
        if self.__pose[2] < -math.pi:
            self.__pose[2] = 2 * math.pi + self.__pose[2]
        elif self.__pose[2] > math.pi:
            self.__pose[2] = self.__pose[2] - 2 * math.pi

        self.__pose[0], self.__pose[1] = self.get_point(
            self.__pose[0], self.__pose[1], self.__pose[2], d)

    def __update_scan(self) -> None:
        angle_list = np.array(self.__SCAN_ANGLES) + self.__pose[2]
        x0 = self.__pose[0]
        y0 = self.__pose[1]

        # Measurement angles must E [-pi,pi].
        for i, _ in enumerate(angle_list):
            if angle_list[i] < -math.pi:
                angle_list[i] = 2 * math.pi + angle_list[i]
            elif angle_list[i] > math.pi:
                angle_list[i] = angle_list[i] - 2 * math.pi

        for i in range(self.__N_MEASUREMENTS):
            x1, y1 = self.get_point(x0, y0, angle_list[i],
                                    self.__SCAN_RANGE_MAX)
            scan_line = ((x0, x1), (y0, y1))
            self.__scan_lines[i] = scan_line

            min_dist = self.__SCAN_RANGE_MAX
            min_x = x1
            min_y = y1
            for wall in self.__track:
                x2 = wall[0][0]
                x3 = wall[0][1]
                y2 = wall[1][0]
                y3 = wall[1][1]

                denominator = (x0 - x1) * (y2 - y3) - (y0 - y1) * (x2 - x3)
                if denominator == 0:
                    continue

                nominator_x = ((x0 * y1 - y0 * x1) * (x2 - x3)
                               - (x0 - x1) * (x2 * y3 - y2 * x3))
                nominator_y = ((x0 * y1 - y0 * x1) * (y2 - y3)
                               - (y0 - y1) * (x2 * y3 - y2 * x3))

                x = nominator_x / denominator
                y = nominator_y / denominator

                not_in_scan_line = ((x1 > x0 and (x < x0 or x > x1)) or
                                    (x0 > x1 and (x < x1 or x > x0)) or
                                    (y1 > y0 and (y < y0 or y > y1)) or
                                    (y0 > y1 and (y < y1 or y > y0)))
                not_in_wall_line = ((x3 > x2 and (x < x2 or x > x3)) or
                                    (x2 > x3 and (x < x3 or x > x2)) or
                                    (y3 > y2 and (y < y2 or y > y3)) or
                                    (y2 > y3 and (y < y3 or y > y2)))
                if not_in_scan_line or not_in_wall_line:
                    continue

                dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_x = x
                    min_y = y

            self.__scan_points[i] = (min_x, min_y)
            sensor_noise = random.gauss(0, self.__SENSOR_STANDARD_DEVIATION)
            min_dist += sensor_noise
            self.__ranges[i] = min_dist

    def __collision_occurred(self) -> bool:
        for range_ in self.__ranges:
            if range_ < self.__WALL_DISTANCE_THRESHOLD:
                return True

        return False

    def __init_pose(self) -> None:
        area = random.choice(self.__spawnable_area)
        self.__pose[0] = random.uniform(area[0][0], area[0][1])
        self.__pose[1] = random.uniform(area[1][0], area[1][1])
        self.__pose[2] = random.uniform(-math.pi, math.pi)

    def reset(self) -> List[float]:
        plt.close()
        plt.figure(figsize=(6.40, 4.80))

        self.__init_pose()

        self.__update_scan()
        observation = list(self.__ranges)

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        assert self.action_space.contains(
            action), f'{action} ({type(action)}) invalid'

        self.__perform_action(action)

        self.__update_scan()
        observation = list(self.__ranges)

        done = self.__collision_occurred()

        if done:
            reward = self.__COLLISION_REWARD
        else:
            if action == self.__FORWARD:
                reward = self.__FORWARD_REWARD
            elif action == self.__YAW_LEFT or action == self.__YAW_RIGHT:
                reward = self.__YAW_REWARD
            else:
                raise ValueError(f'{action} ({type(action)}) invalid')

        return observation, reward, done, []

    def render(self, mode='human') -> None:
        plt.clf()
        for wall in self.__track:
            plt.plot(wall[0], wall[1], 'b')

        for scan_line in self.__scan_lines:
            plt.plot(scan_line[0], scan_line[1], 'y')

        for scan_point in self.__scan_points:
            plt.plot(scan_point[0], scan_point[1], 'co')

        plt.plot(self.__pose[0], self.__pose[1], 'ro')

        plt.xlim((-12, 12))
        plt.ylim((-12, 12))

        plt.pause(0.01)

    def close(self) -> None:
        plt.close()
