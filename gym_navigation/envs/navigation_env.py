import copy
import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, Env

from gym_navigation.utils.line import Line, NoIntersection
from gym_navigation.utils.point import Point
from gym_navigation.utils.pose import Pose


class NavigationEnv(Env):
    metadata = {'render.modes': ['human']}

    __N_ACTIONS = 3
    __FORWARD = 0
    __YAW_RIGHT = 1
    __YAW_LEFT = 2

    __FORWARD_LINEAR_SHIFT = 0.2  # m
    __YAW_LINEAR_SHIFT = 0.04  # m
    __YAW_ANGULAR_SHIFT = 0.2  # rad

    __SHIFT_STANDARD_DEVIATION = 0.02
    __SENSOR_STANDARD_DEVIATION = 0.02

    __COLLISION_THRESHOLD = 0.4

    __COLLISION_REWARD = -200
    __FORWARD_REWARD = +5
    __YAW_REWARD = -0.5

    __SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)
    __SCAN_RANGE_MAX = 30.0
    __SCAN_RANGE_MIN = 0.2
    __N_MEASUREMENTS = len(__SCAN_ANGLES)

    __RENDER_PAUSE_TIME = 0.01
    __Y_LIM = (-12, 12)
    __X_LIM = (-12, 12)

    __TRACK1 = (
        Line(Point(-10, -10), Point(-10, 10)),
        Line(Point(-10, 10), Point(10, 10)),
        Line(Point(10, 10), Point(10, -1.5)),
        Line(Point(10, -1.5), Point(1.5, -1.5)),
        Line(Point(1.5, -1.5), Point(1.5, -10)),
        Line(Point(1.5, -10), Point(-10, -10)),

        Line(Point(-7, -7), Point(-7, 7)),
        Line(Point(-7, 7), Point(7, 7)),
        Line(Point(7, 7), Point(7, 1.5)),
        Line(Point(7, 1.5), Point(-1.5, 1.5)),
        Line(Point(-1.5, 1.5), Point(-1.5, -7)),
        Line(Point(-1.5, -7), Point(-7, -7))
    )

    __TRACKS = (__TRACK1,)

    __SPAWN_AREA1 = (
        ((-8.5, -8.5), (-8.5, 8.5)),
        ((-8.5, 8.5), (8.5, 8.5)),
        ((8.5, 8.5), (0, 8.5)),
        ((0, 8.5), (0, 0)),
        ((0, 0), (-8.5, 0)),
        ((-8.5, 0), (-8.5, -8.5))
    )

    __SPAWN_AREAS = (__SPAWN_AREA1,)

    __track: Tuple[Line, ...]
    __spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    __ranges: np.ndarray
    __pose: Pose
    __scans: np.ndarray
    __scan_intersections: np.ndarray
    __action_space: spaces.Discrete
    __observation_space: spaces.Box

    def __init__(self, track_id: int = 1) -> None:
        if track_id in range(1, len(self.__TRACKS) + 1):
            self.__track = self.__TRACKS[track_id - 1]
            self.__spawn_area = self.__SPAWN_AREAS[track_id - 1]
        else:
            raise ValueError(f'Invalid track id {track_id} ({type(track_id)})')

        self.__ranges = np.empty(self.__N_MEASUREMENTS)

        self.__scans = np.empty(self.__N_MEASUREMENTS, dtype=Line)
        self.__scan_intersections = np.empty(self.__N_MEASUREMENTS,
                                             dtype=Point)

        self.__action_space = spaces.Discrete(self.__N_ACTIONS)

        self.__observation_space = spaces.Box(low=self.__SCAN_RANGE_MAX,
                                              high=self.__SCAN_RANGE_MIN,
                                              shape=(self.__N_MEASUREMENTS,),
                                              dtype=np.float32)

    def __init_pose(self) -> None:
        area = random.choice(self.__spawn_area)
        x = random.uniform(area[0][0], area[0][1])
        y = random.uniform(area[1][0], area[1][1])
        position = Point(x, y)

        yaw = random.uniform(-math.pi, math.pi)
        self.__pose = Pose(position, yaw)

    def __perform_action(self, action: int) -> None:
        theta = random.gauss(0, self.__SHIFT_STANDARD_DEVIATION)
        d = random.gauss(0, self.__SHIFT_STANDARD_DEVIATION)

        if action == self.__FORWARD:
            d += self.__FORWARD_LINEAR_SHIFT
        elif action == self.__YAW_RIGHT:
            d += self.__YAW_LINEAR_SHIFT
            theta += self.__YAW_ANGULAR_SHIFT
        elif action == self.__YAW_LEFT:
            d += self.__YAW_LINEAR_SHIFT
            theta -= self.__YAW_ANGULAR_SHIFT

        self.__pose.shift(d, theta)

    def __update_scan(self) -> None:
        scan_poses = np.empty(self.__N_MEASUREMENTS, dtype=Pose)

        for i, scan_angle in enumerate(self.__SCAN_ANGLES):
            scan_poses[i] = Pose(copy.copy(self.__pose.position),
                                 self.__pose.yaw + scan_angle)

        for i, scan_pose in enumerate(scan_poses):
            closest_intersection_pose = copy.deepcopy(scan_pose)
            closest_intersection_pose.move(self.__SCAN_RANGE_MAX)

            self.__scans[i] = Line(
                copy.copy(scan_pose.position),
                copy.copy(closest_intersection_pose.position))

            min_distance = self.__SCAN_RANGE_MAX
            for wall in self.__track:
                try:
                    intersection = self.__scans[i].get_intersection(wall)
                except NoIntersection:
                    continue

                distance = scan_pose.position.calculate_distance(intersection)
                if distance < min_distance:
                    closest_intersection_pose.position = intersection
                    min_distance = distance

            sensor_noise = random.gauss(0, self.__SENSOR_STANDARD_DEVIATION)
            closest_intersection_pose.move(sensor_noise)
            self.__scan_intersections[i] = closest_intersection_pose.position
            self.__ranges[i] = min_distance + sensor_noise

    def __collision_occurred(self) -> bool:
        return (self.__ranges < self.__COLLISION_THRESHOLD).any()

    def reset(self) -> List[float]:
        plt.close()

        self.__init_pose()

        self.__update_scan()
        observation = list(self.__ranges)

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self.__perform_action(action)

        self.__update_scan()
        observation = list(self.__ranges.copy())

        done = self.__collision_occurred()
        reward = 0

        if done:
            reward = self.__COLLISION_REWARD
        elif action == self.__FORWARD:
            reward = self.__FORWARD_REWARD
        elif action == self.__YAW_LEFT or action == self.__YAW_RIGHT:
            reward = self.__YAW_REWARD

        return observation, reward, done, []

    def render(self, mode='human') -> None:
        if mode != 'human':
            super().render(mode=mode)

        plt.clf()

        for wall in self.__track:
            x_range = (wall.start.x, wall.end.x)
            y_range = (wall.start.y, wall.end.y)
            plt.plot(x_range, y_range, 'b')

        for scan in self.__scans:
            x_range = (scan.start.x, scan.end.x)
            y_range = (scan.start.y, scan.end.y)
            plt.plot(x_range, y_range, 'y')

        for scan_intersection in self.__scan_intersections:
            plt.plot(scan_intersection.x, scan_intersection.y, 'co')

        plt.plot(self.__pose.position.x, self.__pose.position.y, 'ro')

        plt.xlim(self.__X_LIM)
        plt.ylim(self.__Y_LIM)

        plt.pause(self.__RENDER_PAUSE_TIME)

    def close(self) -> None:
        plt.close()

    def seed(self, seed=None):
        return

    @property
    def action_space(self) -> spaces.Discrete:
        return self.__action_space

    @property
    def observation(self) -> spaces.Box:
        return self.__observation_space
