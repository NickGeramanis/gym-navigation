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


class NavigationGoalEnv(Env):
    __metadata = {'render.modes': ['human']}

    __N_ACTIONS = 3
    __FORWARD = 0
    __YAW_RIGHT = 1
    __YAW_LEFT = 2

    __FORWARD_LINEAR_SHIFT = 0.2  # m
    __YAW_LINEAR_SHIFT = 0.04  # m
    __YAW_ANGULAR_SHIFT = 0.2  # rad

    __SHIFT_STANDARD_DEVIATION = 0.02
    __SENSOR_STANDARD_DEVIATION = 0.01

    __COLLISION_THRESHOLD = 0.4
    __GOAL_THRESHOLD = 0.4
    __MINIMUM_DISTANCE = 2

    __TRANSITION_REWARD_FACTOR = 10
    __GOAL_REWARD = 200
    __COLLISION_REWARD = -200

    __SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)
    __SCAN_RANGE_MAX = 30.0
    __SCAN_RANGE_MIN = 0.2
    __N_MEASUREMENTS = len(__SCAN_ANGLES)
    __MAXIMUM_GOAL_DISTANCE = math.inf
    __N_OBSERVATIONS = __N_MEASUREMENTS + 2

    __N_OBSTACLES = 20
    __OBSTACLES_LENGTH = 1

    __RENDER_PAUSE_TIME = 0.01
    __Y_LIM = (-12, 12)
    __X_LIM = (-12, 12)

    __TRACK1 = (
        Line(Point(-10, -10), Point(-10, 10)),
        Line(Point(-10, 10), Point(10, 10)),
        Line(Point(10, 10), Point(10, -10)),
        Line(Point(10, -10), Point(-10, -10))
    )

    __TRACKS = (__TRACK1,)

    __SPAWN_AREA1 = (
        ((-9, 9), (-9, 9)),
    )

    __SPAWN_AREAS = (__SPAWN_AREA1,)

    __track: Tuple[Line, ...]
    __spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    __ranges: np.ndarray
    __pose: Pose
    __goal: Point
    __obstacles: np.ndarray
    __obstacles_centers: np.ndarray
    __distance_from_goal: float
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

        self.__obstacles = np.empty((self.__N_OBSTACLES, 4), dtype=Line)
        self.__obstacles_centers = np.empty(self.__N_OBSTACLES, dtype=Point)
        self.__distance_from_goal = 0.0

        self.__scans = np.empty(self.__N_MEASUREMENTS, dtype=Line)
        self.__scan_intersections = np.empty(self.__N_MEASUREMENTS,
                                             dtype=Point)

        self.__action_space = spaces.Discrete(self.__N_ACTIONS)

        high = np.array(self.__N_MEASUREMENTS * [self.__SCAN_RANGE_MAX]
                        + [self.__MAXIMUM_GOAL_DISTANCE]
                        + [math.pi],
                        dtype=np.float32)

        low = np.array(self.__N_MEASUREMENTS * [self.__SCAN_RANGE_MIN]
                       + [0]
                       + [-math.pi],
                       dtype=np.float32)

        self.__observation_space = spaces.Box(low=low, high=high,
                                              shape=(self.__N_OBSERVATIONS,),
                                              dtype=np.float32)

    def __init_obstacles(self) -> None:
        # Don't check for overlapping obstacles
        # in order to create strange shapes.
        for i in range(self.__N_OBSTACLES):
            area = random.choice(self.__spawn_area)
            x = random.uniform(area[0][0], area[0][1])
            y = random.uniform(area[1][0], area[1][1])
            obstacles_center = Point(x, y)

            self.__obstacles_centers[i] = obstacles_center

            point1 = Point(obstacles_center.x - self.__OBSTACLES_LENGTH / 2,
                           obstacles_center.y - self.__OBSTACLES_LENGTH / 2)
            point2 = Point(obstacles_center.x - self.__OBSTACLES_LENGTH / 2,
                           obstacles_center.y + self.__OBSTACLES_LENGTH / 2)
            point3 = Point(obstacles_center.x + self.__OBSTACLES_LENGTH / 2,
                           obstacles_center.y + self.__OBSTACLES_LENGTH / 2)
            point4 = Point(obstacles_center.x + self.__OBSTACLES_LENGTH / 2,
                           obstacles_center.y - self.__OBSTACLES_LENGTH / 2)

            self.__obstacles[i, 0] = Line(point1, point2)
            self.__obstacles[i, 1] = Line(point2, point3)
            self.__obstacles[i, 2] = Line(point3, point4)
            self.__obstacles[i, 3] = Line(point4, point1)

    def __init_goal(self) -> None:
        done = False
        while not done:
            area = random.choice(self.__spawn_area)
            x = random.uniform(area[0][0], area[0][1])
            y = random.uniform(area[1][0], area[1][1])
            goal = Point(x, y)
            done = True

            for obstacles_center in self.__obstacles_centers:
                distance_from_obstacle = goal.calculate_distance(
                    obstacles_center)
                if distance_from_obstacle < self.__MINIMUM_DISTANCE:
                    done = False

        self.__goal = goal

    def __init_pose(self) -> None:
        done = False
        while not done:
            area = random.choice(self.__spawn_area)
            x = random.uniform(area[0][0], area[0][1])
            y = random.uniform(area[1][0], area[1][1])
            position = Point(x, y)
            done = True

            for obstacles_center in self.__obstacles_centers:
                distance_from_obstacle = position.calculate_distance(
                    obstacles_center)
                if distance_from_obstacle < self.__MINIMUM_DISTANCE:
                    done = False

            distance_from_goal = position.calculate_distance(self.__goal)
            if distance_from_goal < self.__MINIMUM_DISTANCE:
                done = False

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
            walls = np.concatenate((self.__track, self.__obstacles.flatten()))
            for wall in walls:
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
        return bool((self.__ranges < self.__COLLISION_THRESHOLD).any())

    def reset(self) -> List[float]:
        plt.close()

        self.__init_obstacles()
        self.__init_goal()
        self.__init_pose()

        self.__update_scan()

        distance_from_goal = self.__pose.position.calculate_distance(
            self.__goal)
        angle_from_goal = self.__pose.calculate_angle_difference(self.__goal)

        observation = list(self.__ranges)
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        self.__distance_from_goal = distance_from_goal

        return observation

    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self.__perform_action(action)

        self.__update_scan()

        distance_from_goal = self.__pose.position.calculate_distance(
            self.__goal)
        angle_from_goal = self.__pose.calculate_angle_difference(self.__goal)

        observation = list(self.__ranges)
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        reward: float
        if self.__collision_occurred():
            reward = self.__COLLISION_REWARD
            done = True
        elif distance_from_goal < self.__GOAL_THRESHOLD:
            reward = self.__GOAL_REWARD
            done = True
        else:
            reward = (self.__TRANSITION_REWARD_FACTOR
                      * (self.__distance_from_goal - distance_from_goal))
            done = False

        self.__distance_from_goal = distance_from_goal

        return observation, reward, done, []

    def render(self, mode='human') -> None:
        if mode != 'human':
            super().render(mode=mode)

        plt.clf()

        walls = np.concatenate((self.__track, self.__obstacles.flatten()))
        for wall in walls:
            x_range = (wall.start.x, wall.end.x)
            y_range = (wall.start.y, wall.end.y)
            plt.plot(x_range, y_range, 'b')

        for scan in self.__scans:
            x_range = (scan.start.x, scan.end.x)
            y_range = (scan.start.y, scan.end.y)
            plt.plot(x_range, y_range, 'y')

        for __scan_intersection in self.__scan_intersections:
            plt.plot(__scan_intersection.x, __scan_intersection.y, 'co')

        plt.plot(self.__pose.position.x, self.__pose.position.y, 'ro')
        plt.plot(self.__goal.x, self.__goal.y, 'go')

        plt.xlim(self.__X_LIM)
        plt.ylim(self.__Y_LIM)

        plt.pause(self.__RENDER_PAUSE_TIME)

    def close(self) -> None:
        plt.close()

    @property
    def action_space(self) -> spaces.Discrete:
        return self.__action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self.__observation_space

    @property
    def metadata(self) -> Dict:
        return self.__metadata
