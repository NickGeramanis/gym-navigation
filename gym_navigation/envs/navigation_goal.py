"""This module contains the Navigation Goal environment class."""
import math
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Box
from pygame import Surface

from gym_navigation.enums.color import Color
from gym_navigation.envs.navigation_track import NavigationTrack
from gym_navigation.geometry.line import Line
from gym_navigation.geometry.point import Point


class NavigationGoal(NavigationTrack):
    """The Navigation Goal environment."""
    _GOAL_THRESHOLD = 0.4
    _MINIMUM_DISTANCE = 3

    _ANGLE_STANDARD_DEVIATION = 0.02
    _DISTANCE_STANDARD_DEVIATION = 0.02

    _TRANSITION_REWARD_FACTOR = 10
    _GOAL_REWARD = 200.0

    _MAXIMUM_GOAL_DISTANCE = 100
    _N_OBSERVATIONS = NavigationTrack._N_MEASUREMENTS + 2

    _N_OBSTACLES = 20
    _OBSTACLES_LENGTH = 1

    _distance_from_goal: float
    _previous_distance_from_goal: float
    _goal: Point

    def __init__(self,
                 render_mode: Optional[str] = None,
                 track_id: int = 2) -> None:
        super().__init__(render_mode, track_id)

        high = np.array(self._N_MEASUREMENTS * [self._SCAN_RANGE_MAX]
                        + [self._MAXIMUM_GOAL_DISTANCE]
                        + [math.pi],
                        dtype=np.float64)

        low = np.array(self._N_MEASUREMENTS * [self._SCAN_RANGE_MIN]
                       + [0.0]
                       + [-math.pi],
                       dtype=np.float64)

        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=(self._N_OBSERVATIONS,),
                                     dtype=np.float64)

    def _do_perform_action(self, action: int) -> None:
        super()._do_perform_action(action)
        self._distance_from_goal = (
            self._DISTANCE_STANDARD_DEVIATION
            + self._pose.position.calculate_distance(self._goal))

    def _do_get_observation(self) -> np.ndarray:
        angle_from_goal = (self._ANGLE_STANDARD_DEVIATION
                           + self._pose.calculate_angle_difference(self._goal))
        return np.append(
            self._ranges.copy(),
            [self._distance_from_goal, angle_from_goal])

    def _do_check_if_terminated(self) -> bool:
        return self._collision_occurred() or self._goal_reached()

    def _goal_reached(self) -> bool:
        return self._distance_from_goal < self._GOAL_THRESHOLD

    def _do_calculate_reward(self, action: int) -> float:
        if self._collision_occurred():
            reward = self._COLLISION_REWARD
        elif self._goal_reached():
            reward = self._GOAL_REWARD
        else:
            reward = (
                self._TRANSITION_REWARD_FACTOR
                * (self._previous_distance_from_goal -
                   self._distance_from_goal))

        self._previous_distance_from_goal = self._distance_from_goal
        return reward

    def _do_init_environment(self, options: Optional[dict] = None) -> None:
        self._init_pose()
        self._init_goal()
        self._init_obstacles()
        self._update_scan()

    def _init_goal(self) -> None:
        while True:
            area = self.np_random.choice(self._track.spawn_area)
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            goal = Point(x_coordinate, y_coordinate)
            distance_from_pose = goal.calculate_distance(self._pose.position)
            if distance_from_pose > self._MINIMUM_DISTANCE:
                break

        self._goal = goal
        self._distance_from_goal = self._pose.position.calculate_distance(
            self._goal)
        self._previous_distance_from_goal = self._distance_from_goal

    def _init_obstacles(self) -> None:
        for _ in range(self._N_OBSTACLES):
            while True:
                area = self.np_random.choice(self._track.spawn_area)
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                obstacles_center = Point(x_coordinate, y_coordinate)
                distance_from_pose = obstacles_center.calculate_distance(
                    self._pose.position)
                distance_from_goal = obstacles_center.calculate_distance(
                    self._goal)

                if (distance_from_pose > self._MINIMUM_DISTANCE
                        and distance_from_goal > self._MINIMUM_DISTANCE):
                    break

            point1 = Point(
                obstacles_center.x_coordinate - self._OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate - self._OBSTACLES_LENGTH / 2)
            point2 = Point(
                obstacles_center.x_coordinate - self._OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate + self._OBSTACLES_LENGTH / 2)
            point3 = Point(
                obstacles_center.x_coordinate + self._OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate + self._OBSTACLES_LENGTH / 2)
            point4 = Point(
                obstacles_center.x_coordinate + self._OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate - self._OBSTACLES_LENGTH / 2)

            self._world += (Line(point1, point2),)
            self._world += (Line(point2, point3),)
            self._world += (Line(point3, point4),)
            self._world += (Line(point4, point1),)

    def _do_draw(self, canvas: Surface) -> None:
        super()._do_draw(canvas)
        pygame.draw.circle(canvas,
                           Color.GREEN.value,
                           self._convert_point(self._goal),
                           self._GOAL_THRESHOLD * self._RESOLUTION)
