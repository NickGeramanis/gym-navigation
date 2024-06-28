"""This module contains the Navigation Track environment class."""
import copy
import math
from typing import Optional, Tuple

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pygame import Surface

from gym_navigation.enums.action import Action
from gym_navigation.enums.color import Color
from gym_navigation.envs.navigation import Navigation
from gym_navigation.geometry.line import Line, NoIntersectionError
from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose


class NavigationTrack(Navigation):
    """The Navigation Track environment."""
    _SHIFT_STANDARD_DEVIATION = 0.02
    _SENSOR_STANDARD_DEVIATION = 0.02

    _COLLISION_THRESHOLD = 0.4

    _COLLISION_REWARD = -200.0
    _FORWARD_REWARD = +5.0
    _ROTATION_REWARD = -0.5

    _SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)
    _SCAN_RANGE_MAX = 30.0
    _SCAN_RANGE_MIN = 0.2
    _N_MEASUREMENTS = len(_SCAN_ANGLES)
    _N_OBSERVATIONS = _N_MEASUREMENTS

    _pose: Pose
    _ranges: np.ndarray

    def __init__(self,
                 render_mode: Optional[str] = None,
                 track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        self._ranges = np.empty(self._N_MEASUREMENTS)

        self.action_space = Discrete(len(Action))

        self.observation_space = Box(low=self._SCAN_RANGE_MIN,
                                     high=self._SCAN_RANGE_MAX,
                                     shape=(self._N_OBSERVATIONS,),
                                     dtype=np.float64)

    def _do_perform_action(self, action: int) -> None:
        action_enum = Action(action)
        theta = (self.np_random.normal(0, self._SHIFT_STANDARD_DEVIATION)
                 + action_enum.angular_shift)
        distance = (self.np_random.normal(0, self._SHIFT_STANDARD_DEVIATION)
                    + action_enum.linear_shift)
        self._pose.shift(distance, theta)
        self._update_scan()

    def _update_scan(self) -> None:
        scan_lines = self._create_scan_lines()
        for i, scan_line in enumerate(scan_lines):
            min_distance = self._SCAN_RANGE_MAX
            for wall in self._world:
                try:
                    intersection = scan_line.get_intersection(wall)
                except NoIntersectionError:
                    continue

                distance = self._pose.position.calculate_distance(intersection)
                min_distance = min(min_distance, distance)

            sensor_noise = self.np_random.normal(
                0, self._SENSOR_STANDARD_DEVIATION)
            self._ranges[i] = min_distance + sensor_noise

    def _create_scan_lines(self) -> np.ndarray:
        scan_poses = self._create_scan_poses()
        scan_lines = np.empty(self._N_MEASUREMENTS, dtype=Line)

        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self._SCAN_RANGE_MAX)
            scan_lines[i] = Line(copy.copy(self._pose.position),
                                 scan_pose.position)

        return scan_lines

    def _create_scan_poses(self) -> np.ndarray:
        scan_poses = np.empty(self._N_MEASUREMENTS, dtype=Pose)

        for i, scan_angle in enumerate(self._SCAN_ANGLES):
            scan_poses[i] = Pose(copy.copy(self._pose.position),
                                 self._pose.yaw + scan_angle)

        return scan_poses

    def _do_get_observation(self) -> np.ndarray:
        return self._ranges.copy()

    def _do_check_if_terminated(self) -> bool:
        return self._collision_occurred()

    def _collision_occurred(self) -> bool:
        return bool((self._ranges < self._COLLISION_THRESHOLD).any())

    def _do_calculate_reward(self, action: int) -> float:
        if self._collision_occurred():
            reward = self._COLLISION_REWARD
        elif action == Action.FORWARD.value:
            reward = self._FORWARD_REWARD
        else:
            reward = self._ROTATION_REWARD

        return reward

    def _do_init_environment(self, options: Optional[dict] = None) -> None:
        self._init_pose()
        self._update_scan()

    def _do_create_info(self) -> dict:
        return {}

    def _init_pose(self) -> None:
        area = self.np_random.choice(self._track.spawn_area)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        position = Point(x_coordinate, y_coordinate)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        self._pose = Pose(position, yaw)

    def _do_draw(self, canvas: Surface) -> None:
        canvas.fill(Color.WHITE.value)

        for wall in self._world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self._convert_point(wall.start),
                             self._convert_point(wall.end),
                             self._WIDTH)

        scan_poses = self._create_scan_poses()
        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self._ranges[i])
            pygame.draw.line(canvas,
                             Color.RED.value,
                             self._convert_point(self._pose.position),
                             self._convert_point(scan_pose.position),
                             self._WIDTH)

        pygame.draw.circle(canvas,
                           Color.BLUE.value,
                           self._convert_point(self._pose.position),
                           self._COLLISION_THRESHOLD * self._RESOLUTION)

    def _convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self._RESOLUTION)
                    + self._X_OFFSET)
        pygame_y = (self._WINDOW_SIZE
                    - round(point.y_coordinate * self._RESOLUTION)
                    + self._Y_OFFSET)
        return pygame_x, pygame_y
