"""This Module contains the basic Navigation environment class."""
import copy
import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces

from gym_navigation.utils.line import Line, NoIntersection
from gym_navigation.utils.point import Point
from gym_navigation.utils.pose import Pose


class Navigation(Env):
    """The basic Navigation environment."""
    _metadata = {'render.modes': ['human']}

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

    _COLLISION_REWARD = -200
    _FORWARD_REWARD = +5
    _YAW_REWARD = -0.5

    _SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)
    _SCAN_RANGE_MAX = 30.0
    _SCAN_RANGE_MIN = 0.2
    _N_MEASUREMENTS = len(_SCAN_ANGLES)
    _N_OBSERVATIONS = _N_MEASUREMENTS

    _RENDER_PAUSE_TIME = 0.01

    _Y_LIM = (-12, 12)
    _X_LIM = (-12, 12)

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

    _track: Tuple[Line, ...]
    _spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    _ranges: np.ndarray
    _pose: Pose
    _scans: np.ndarray
    _scan_intersections: np.ndarray
    _action_space: spaces.Discrete
    _observation_space: spaces.Box

    def __init__(self, track_id: int = 1) -> None:
        if track_id in range(1, len(self.__TRACKS) + 1):
            self._track = self.__TRACKS[track_id - 1]
            self._spawn_area = self.__SPAWN_AREAS[track_id - 1]
        else:
            raise ValueError(f'Invalid track id {track_id} ({type(track_id)})')

        self._ranges = np.empty(self._N_MEASUREMENTS)

        self._scans = np.empty(self._N_MEASUREMENTS, dtype=Line)
        self._scan_intersections = np.empty(self._N_MEASUREMENTS,
                                            dtype=Point)

        self._action_space = spaces.Discrete(self._N_ACTIONS)

        self._observation_space = spaces.Box(low=self._SCAN_RANGE_MAX,
                                             high=self._SCAN_RANGE_MIN,
                                             shape=(self._N_OBSERVATIONS,),
                                             dtype=np.float32)

    def _init_pose(self) -> None:
        area = random.choice(self._spawn_area)
        x = random.uniform(area[0][0], area[0][1])
        y = random.uniform(area[1][0], area[1][1])
        position = Point(x, y)

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
        elif action == self._YAW_LEFT:
            distance += self._YAW_LINEAR_SHIFT
            theta -= self._YAW_ANGULAR_SHIFT

        self._pose.shift(distance, theta)

    def _update_scan(self) -> None:
        scan_poses = np.empty(self._N_MEASUREMENTS, dtype=Pose)

        for i, scan_angle in enumerate(self._SCAN_ANGLES):
            scan_poses[i] = Pose(copy.copy(self._pose.position),
                                 self._pose.yaw + scan_angle)

        for i, scan_pose in enumerate(scan_poses):
            closest_intersection_pose = copy.deepcopy(scan_pose)
            closest_intersection_pose.move(self._SCAN_RANGE_MAX)

            self._scans[i] = Line(
                copy.copy(scan_pose.position),
                copy.copy(closest_intersection_pose.position))

            min_distance = self._SCAN_RANGE_MAX
            for wall in self._track:
                try:
                    intersection = self._scans[i].get_intersection(wall)
                except NoIntersection:
                    continue

                distance = scan_pose.position.calculate_distance(intersection)
                if distance < min_distance:
                    closest_intersection_pose.position = intersection
                    min_distance = distance

            sensor_noise = random.gauss(0, self._SENSOR_STANDARD_DEVIATION)
            closest_intersection_pose.move(sensor_noise)
            self._scan_intersections[i] = closest_intersection_pose.position
            self._ranges[i] = min_distance + sensor_noise

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
        reward = 0.0

        if done:
            reward = self._COLLISION_REWARD
        elif action == self._FORWARD:
            reward = self._FORWARD_REWARD
        elif action in (self._YAW_LEFT, self._YAW_RIGHT):
            reward = self._YAW_REWARD

        return observation, reward, done, []

    def render(self, mode: str = 'human') -> None:
        if mode not in self._metadata['render.modes']:
            raise ValueError('Mode {mode} is not supported')

        self._plot()
        plt.pause(self._RENDER_PAUSE_TIME)

    def _plot(self):
        plt.clf()

        for wall in self._track:
            x_range = (wall.start.x, wall.end.x)
            y_range = (wall.start.y, wall.end.y)
            plt.plot(x_range, y_range, 'b')

        for scan in self._scans:
            x_range = (scan.start.x, scan.end.x)
            y_range = (scan.start.y, scan.end.y)
            plt.plot(x_range, y_range, 'y')

        for scan_intersection in self._scan_intersections:
            plt.plot(scan_intersection.x, scan_intersection.y, 'co')

        plt.plot(self._pose.position.x, self._pose.position.y, 'ro')

        plt.xlim(self._X_LIM)
        plt.ylim(self._Y_LIM)

    def close(self) -> None:
        plt.close()

    @property
    def action_space(self) -> spaces.Discrete:
        """The action space of the environemt."""
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        """The observation space of the environemt."""
        return self._observation_space

    @property
    def metadata(self) -> Dict:
        """The metadata of the environemt."""
        return self._metadata
