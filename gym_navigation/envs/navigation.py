"""This module contains the basic Navigation environment class."""
import copy
import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces

from gym_navigation.utils.line import Line, NoIntersectionError
from gym_navigation.utils.point import Point
from gym_navigation.utils.pose import Pose


class Navigation(Env):
    """The basic Navigation environment."""
    metadata = {'render.modes': ['human']}

    _N_ACTIONS = 3
    _FORWARD = 0
    _YAW_RIGHT = 1
    _YAW_LEFT = 2

    _FORWARD_LINEAR_SHIFT = 0.2  # m
    _YAW_LINEAR_SHIFT = 0.04  # m
    _YAW_ANGULAR_SHIFT = 0.2  # rad

    _SHIFT_STANDARD_DEVIATION = 0.02
    _SENSOR_STANDARD_DEVIATION = 0.02

    _COLLISION_THRESHOLD = 0.4

    _COLLISION_REWARD = -200.0
    _FORWARD_REWARD = +5.0
    _YAW_REWARD = -0.5

    _SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)
    _SCAN_RANGE_MAX = 30.0
    _SCAN_RANGE_MIN = 0.2
    _N_MEASUREMENTS = len(_SCAN_ANGLES)
    _N_OBSERVATIONS = _N_MEASUREMENTS

    _RENDER_PAUSE_TIME = 0.01

    _Y_LIM = (-12, 12)
    _X_LIM = (-12, 12)

    _TRACK1: Tuple[Line, ...] = (
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

    _TRACKS = (_TRACK1,)

    _SPAWN_AREA1: Tuple[
        Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
        ((-8.5, -8.5), (-8.5, 8.5)),
        ((-8.5, 8.5), (8.5, 8.5)),
        ((8.5, 8.5), (0, 8.5)),
        ((0, 8.5), (0, 0)),
        ((0, 0), (-8.5, 0)),
        ((-8.5, 0), (-8.5, -8.5))
    )

    _SPAWN_AREAS = (_SPAWN_AREA1,)

    _track: Tuple[Line, ...]
    _spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    _ranges: np.ndarray
    _pose: Pose
    action_space: spaces.Discrete
    observation_space: spaces.Box

    def __init__(self, track_id: int = 1) -> None:
        if track_id in range(1, len(self._TRACKS) + 1):
            self._track = self._TRACKS[track_id - 1]
            self._spawn_area = self._SPAWN_AREAS[track_id - 1]
        else:
            raise ValueError(f'Invalid track id {track_id} ({type(track_id)})')

        self._ranges = np.empty(self._N_MEASUREMENTS)

        self.action_space = spaces.Discrete(self._N_ACTIONS)

        self.observation_space = spaces.Box(low=self._SCAN_RANGE_MAX,
                                            high=self._SCAN_RANGE_MIN,
                                            shape=(self._N_OBSERVATIONS,),
                                            dtype=np.float32)

    def _init_pose(self) -> None:
        area = random.choice(self._spawn_area)
        x_coordinate = random.uniform(area[0][0], area[0][1])
        y_coordinate = random.uniform(area[1][0], area[1][1])
        position = Point(x_coordinate, y_coordinate)
        yaw = random.uniform(-math.pi, math.pi)
        self._pose = Pose(position, yaw)

    def _perform_action(self, action: int) -> None:
        theta = random.gauss(0, self._SHIFT_STANDARD_DEVIATION)
        distance = random.gauss(0, self._SHIFT_STANDARD_DEVIATION)

        if action == self._FORWARD:
            distance += self._FORWARD_LINEAR_SHIFT
        elif action == self._YAW_RIGHT:
            distance += self._YAW_LINEAR_SHIFT
            theta += self._YAW_ANGULAR_SHIFT
        else:
            distance += self._YAW_LINEAR_SHIFT
            theta -= self._YAW_ANGULAR_SHIFT

        self._pose.shift(distance, theta)

    def _update_scan(self) -> None:
        scan_lines = self._create_scan_lines()
        for i, scan_line in enumerate(scan_lines):
            min_distance = self._SCAN_RANGE_MAX
            for wall in self._track:
                try:
                    intersection = scan_line.get_intersection(wall)
                except NoIntersectionError:
                    continue

                distance = self._pose.position.calculate_distance(intersection)
                if distance < min_distance:
                    min_distance = distance

            sensor_noise = random.gauss(0, self._SENSOR_STANDARD_DEVIATION)
            self._ranges[i] = min_distance + sensor_noise

    def _create_scan_poses(self) -> np.ndarray:
        scan_poses = np.empty(self._N_MEASUREMENTS, dtype=Pose)

        for i, scan_angle in enumerate(self._SCAN_ANGLES):
            scan_poses[i] = Pose(copy.copy(self._pose.position),
                                 self._pose.yaw + scan_angle)

        return scan_poses

    def _create_scan_lines(self) -> np.ndarray:
        scan_poses = self._create_scan_poses()
        scan_lines = np.empty(self._N_MEASUREMENTS, dtype=Line)

        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self._SCAN_RANGE_MAX)
            scan_lines[i] = Line(copy.copy(self._pose.position),
                                 scan_pose.position)

        return scan_lines

    def _collision_occurred(self) -> bool:
        return bool((self._ranges < self._COLLISION_THRESHOLD).any())

    def reset(self) -> List[float]:
        plt.close()

        self._init_pose()

        self._update_scan()
        observation = list(self._ranges)

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._perform_action(action)

        self._update_scan()
        observation = list(self._ranges)

        done = self._collision_occurred()

        if done:
            reward = self._COLLISION_REWARD
        elif action == self._FORWARD:
            reward = self._FORWARD_REWARD
        else:
            reward = self._YAW_REWARD

        return observation, reward, done, []

    def render(self, mode: str = 'human') -> None:
        if mode not in self.metadata['render.modes']:
            raise ValueError(f'Mode {mode} is not supported')

        self._plot()

        plt.pause(self._RENDER_PAUSE_TIME)

    def _plot(self):
        plt.clf()

        plt.xlim(self._X_LIM)
        plt.ylim(self._Y_LIM)

        for wall in self._track:
            x_range = (wall.start.x_coordinate, wall.end.x_coordinate)
            y_range = (wall.start.y_coordinate, wall.end.y_coordinate)
            plt.plot(x_range, y_range, 'b')

        scan_lines = self._create_scan_lines()
        for scan_line in scan_lines:
            x_range = (scan_line.start.x_coordinate,
                       scan_line.end.x_coordinate)
            y_range = (scan_line.start.y_coordinate,
                       scan_line.end.y_coordinate)
            plt.plot(x_range, y_range, 'y')

        scan_poses = self._create_scan_poses()
        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self._ranges[i])
            plt.plot(scan_pose.position.x_coordinate,
                     scan_pose.position.y_coordinate,
                     'co')

        plt.plot(self._pose.position.x_coordinate,
                 self._pose.position.y_coordinate,
                 'ro')

    def close(self) -> None:
        plt.close()
