"""This module contains the Navigation Goal environment class."""
import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gym_navigation.envs.navigation import Navigation
from gym_navigation.utils.line import Line
from gym_navigation.utils.point import Point


class NavigationGoal(Navigation):
    """The Navigation Goal environment."""
    _GOAL_THRESHOLD = 0.4
    _MINIMUM_DISTANCE = 3

    _TRANSITION_REWARD_FACTOR = 10
    _GOAL_REWARD = 200.0

    _MAXIMUM_GOAL_DISTANCE = math.inf
    _N_OBSERVATIONS = Navigation._N_MEASUREMENTS + 2

    _N_OBSTACLES = 20
    _OBSTACLES_LENGTH = 1

    _TRACK1 = (
        Line(Point(-10, -10), Point(-10, 10)),
        Line(Point(-10, 10), Point(10, 10)),
        Line(Point(10, 10), Point(10, -10)),
        Line(Point(10, -10), Point(-10, -10))
    )

    _TRACKS = (_TRACK1,)

    _SPAWN_AREA1 = (
        ((-9, 9), (-9, 9)),
    )

    _SPAWN_AREAS = (_SPAWN_AREA1,)

    _track_id: int
    _distance_from_goal: float
    _goal: Point

    def __init__(self, track_id: int = 1) -> None:
        super().__init__(track_id)
        self._track_id = track_id

        high = np.array(self._N_MEASUREMENTS * [self._SCAN_RANGE_MAX]
                        + [self._MAXIMUM_GOAL_DISTANCE]
                        + [math.pi],
                        dtype=np.float32)

        low = np.array(self._N_MEASUREMENTS * [self._SCAN_RANGE_MIN]
                       + [0.0]
                       + [-math.pi],
                       dtype=np.float32)

        self._observation_space = spaces.Box(low=low,
                                             high=high,
                                             shape=(self._N_OBSERVATIONS,),
                                             dtype=np.float32)

    def _init_obstacles(self) -> None:
        self._track = self._TRACKS[self._track_id - 1]
        # Don't check for overlapping obstacles
        # in order to create strange shapes.
        for _ in range(self._N_OBSTACLES):
            while True:
                area = random.choice(self._spawn_area)
                x_coordinate = random.uniform(area[0][0], area[0][1])
                y_coordinate = random.uniform(area[1][0], area[1][1])
                obstacles_center = Point(x_coordinate, y_coordinate)
                distance_from_pose = obstacles_center.calculate_distance(
                    self._pose.position)
                distance_from_goal = obstacles_center.calculate_distance(
                    self._goal)

                if (distance_from_pose > self._MINIMUM_DISTANCE
                        or distance_from_goal < self._MINIMUM_DISTANCE):
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

            self._track += (Line(point1, point2),)
            self._track += (Line(point2, point3),)
            self._track += (Line(point3, point4),)
            self._track += (Line(point4, point1),)

    def _init_goal(self) -> None:
        while True:
            area = random.choice(self._spawn_area)
            x_coordinate = random.uniform(area[0][0], area[0][1])
            y_coordinate = random.uniform(area[1][0], area[1][1])
            goal = Point(x_coordinate, y_coordinate)
            distance_from_pose = goal.calculate_distance(self._pose.position)
            if distance_from_pose > self._MINIMUM_DISTANCE:
                break

        self._goal = goal

    def reset(self) -> List[float]:
        plt.close()

        self._init_pose()
        self._init_goal()
        self._init_obstacles()

        self._update_scan()

        distance_from_goal = self._pose.position.calculate_distance(self._goal)
        angle_from_goal = self._pose.calculate_angle_difference(self._goal)

        observation = list(self._ranges)
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        self._distance_from_goal = distance_from_goal

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._perform_action(action)

        self._update_scan()

        distance_from_goal = self._pose.position.calculate_distance(self._goal)
        angle_from_goal = self._pose.calculate_angle_difference(self._goal)

        observation = list(self._ranges)
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        reward: float
        if self._collision_occurred():
            reward = self._COLLISION_REWARD
            done = True
        elif distance_from_goal < self._GOAL_THRESHOLD:
            reward = self._GOAL_REWARD
            done = True
        else:
            reward = (self._TRANSITION_REWARD_FACTOR
                      * (self._distance_from_goal - distance_from_goal))
            done = False

        self._distance_from_goal = distance_from_goal

        return observation, reward, done, []

    def render(self, mode: str = 'human') -> None:
        if mode not in self._metadata['render.modes']:
            raise ValueError(f'Mode {mode} is not supported')

        self._plot()
        plt.plot(self._goal.x_coordinate, self._goal.y_coordinate, 'go')

        plt.pause(self._RENDER_PAUSE_TIME)
