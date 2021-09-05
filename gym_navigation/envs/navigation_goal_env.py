import math
import random
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gym_navigation.envs.navigation_env import NavigationEnv


class NavigationGoalEnv(gym.Env):
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
    __GOAL_DISTANCE_THRESHOLD = 0.4
    __MINIMUM_DISTANCE = 2

    __TRANSITION_REWARD_FACTOR = 10
    __GOAL_REWARD = 200
    __COLLISION_REWARD = -200

    __SCAN_ANGLES = (-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2)

    __SCAN_RANGE_MAX = 30.0
    __SCAN_RANGE_MIN = 0.2
    __N_MEASUREMENTS = len(__SCAN_ANGLES)
    __MAXIMUM_GOAL_DISTANCE = 30
    __N_OBSERVATIONS = __N_MEASUREMENTS + 2
    __N_OBSTACLES = 20
    __OBSTACLES_LENGTH = 1

    __TRACK1 = (
        ((-10, -10), (-10, 10)),
        ((-10, 10), (10, 10)),
        ((10, 10), (10, -10)),
        ((10, -10), (-10, -10))
    )

    __SPAWNABLE_AREA1 = (
        ((-9, 9), (-9, 9)),
    )

    def __init__(self, track_id: int = 1) -> None:
        if track_id == 1:
            self.__track = self.__TRACK1
            self.__spawnable_area = self.__SPAWNABLE_AREA1
        else:
            raise ValueError('Invalid track id')

        self.__ranges = np.empty((self.__N_MEASUREMENTS,))

        # Pose = (x, y, yaw).
        # Note that yaw is measured from the y axis and E [-pi, pi].
        self.__pose = np.empty((3,))
        self.__goal = np.empty((2,))
        self.__obstacles_lines = np.empty((self.__N_OBSTACLES * 4, 2, 2))
        self.__obstacles_centers = np.empty((self.__N_OBSTACLES, 2))
        self.__distance_from_goal = 0.0

        self.__scan_lines = np.empty((self.__N_MEASUREMENTS,), dtype=object)
        self.__scan_points = np.empty((self.__N_MEASUREMENTS,), dtype=object)

        self.action_space = spaces.Discrete(self.__N_ACTIONS)

        high = self.__N_MEASUREMENTS * [self.__SCAN_RANGE_MAX] + [
            self.__MAXIMUM_GOAL_DISTANCE] + [math.pi]
        high = np.array(high, dtype=np.float32)

        low = self.__N_MEASUREMENTS * [self.__SCAN_RANGE_MIN] + [0] + [
            -math.pi]
        low = np.array(low, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low, high=high, shape=(self.__N_OBSERVATIONS,),
            dtype=np.float32)

    def __init_obstacles(self) -> None:
        # Don't check for overlapping obstacles
        # in order to create strange shapes.
        for obstacle_i in range(self.__N_OBSTACLES):
            area = random.choice(self.__spawnable_area)
            obstacle_x = random.uniform(area[0][0], area[0][1])
            obstacle_y = random.uniform(area[1][0], area[1][1])

            self.__obstacles_centers[obstacle_i][0] = obstacle_x
            self.__obstacles_centers[obstacle_i][1] = obstacle_y

            starting_point = (obstacle_x - self.__OBSTACLES_LENGTH / 2,
                              obstacle_y - self.__OBSTACLES_LENGTH / 2)

            line1 = ((starting_point[0], starting_point[0]),
                     (starting_point[1], starting_point[1]
                      + self.__OBSTACLES_LENGTH))
            line2 = ((line1[0][0], line1[0][0] + self.__OBSTACLES_LENGTH),
                     (line1[1][1], line1[1][1]))
            line3 = ((line2[0][1], line2[0][1]),
                     (line2[1][1], line2[1][1] - self.__OBSTACLES_LENGTH))
            line4 = ((line3[0][1], line3[0][1] - self.__OBSTACLES_LENGTH),
                     (line3[1][1], line3[1][1]))

            self.__obstacles_lines[obstacle_i * 4] = line1
            self.__obstacles_lines[obstacle_i * 4 + 1] = line2
            self.__obstacles_lines[obstacle_i * 4 + 2] = line3
            self.__obstacles_lines[obstacle_i * 4 + 3] = line4

    def __init_goal(self) -> None:
        done = False
        goal_x = goal_y = 0
        while not done:
            area = random.choice(self.__spawnable_area)
            goal_x = random.uniform(area[0][0], area[0][1])
            goal_y = random.uniform(area[1][0], area[1][1])
            done = True

            for obstacle in self.__obstacles_centers:
                distance_from_obstacle = np.linalg.norm(
                    np.array([goal_x, goal_y]) - obstacle)
                if distance_from_obstacle < self.__MINIMUM_DISTANCE:
                    done = False

        self.__goal[0] = goal_x
        self.__goal[1] = goal_y

    def __init_pose(self) -> None:
        done = False
        pose_x = pose_y = 0
        while not done:
            area = random.choice(self.__spawnable_area)
            pose_x = random.uniform(area[0][0], area[0][1])
            pose_y = random.uniform(area[1][0], area[1][1])
            done = True

            for obstacle in self.__obstacles_centers:
                distance_from_obstacle = np.linalg.norm(
                    np.array([pose_x, pose_y]) - obstacle)
                if distance_from_obstacle < self.__MINIMUM_DISTANCE:
                    done = False

            distance_from_goal = np.linalg.norm(
                np.array([pose_x, pose_y]) - self.__goal)
            if distance_from_goal < self.__MINIMUM_DISTANCE:
                done = False

        self.__pose[0] = pose_x
        self.__pose[1] = pose_y
        self.__pose[2] = random.uniform(-math.pi, math.pi)

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

        # yaw must E [-pi,pi]
        if self.__pose[2] < -math.pi:
            self.__pose[2] = 2 * math.pi + self.__pose[2]
        elif self.__pose[2] > math.pi:
            self.__pose[2] = self.__pose[2] - 2 * math.pi

        self.__pose[0], self.__pose[1] = NavigationEnv.get_point(
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
            x1, y1 = NavigationEnv.get_point(
                x0, y0, angle_list[i], self.__SCAN_RANGE_MAX)
            scan_line = ((x0, x1), (y0, y1))
            self.__scan_lines[i] = scan_line

            min_dist = self.__SCAN_RANGE_MAX
            min_x = x1
            min_y = y1
            for wall in np.concatenate((self.__track, self.__obstacles_lines),
                                       axis=0):
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

    def __calculate_angle_from_goal(self) -> float:
        goal_angle = math.atan2(
            self.__goal[1] - self.__pose[1], self.__goal[0] - self.__pose[0])
        angle_from_goal = goal_angle - self.__pose[2]
        if angle_from_goal > math.pi:
            angle_from_goal -= 2 * math.pi
        elif angle_from_goal < -math.pi:
            angle_from_goal += 2 * math.pi
        return angle_from_goal

    def __calculate_distance_from_goal(self) -> float:
        position = np.array([self.__pose[0], self.__pose[1]])
        distance_from_goal = np.linalg.norm(position - self.__goal)
        return distance_from_goal

    def reset(self) -> List[float]:
        plt.close()
        plt.figure(figsize=(6.40, 4.80))

        self.__init_obstacles()
        self.__init_goal()
        self.__init_pose()

        self.__update_scan()

        position = np.array([self.__pose[0], self.__pose[1]])
        distance_from_goal = np.linalg.norm(position - self.__goal)

        goal_angle = math.atan2(
            self.__goal[1] - self.__pose[1], self.__goal[0] - self.__pose[0])
        angle_from_goal = goal_angle - self.__pose[2]
        if angle_from_goal > math.pi:
            angle_from_goal -= 2 * math.pi
        elif angle_from_goal < -math.pi:
            angle_from_goal += 2 * math.pi

        observation = list(self.__ranges[:])
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        self.__distance_from_goal = distance_from_goal

        return observation

    def step(self, action: int) -> Tuple[List[float], int, bool, List[str]]:
        assert self.action_space.contains(
            action), f'{action} ({type(action)}) invalid'

        self.__perform_action(action)

        self.__update_scan()

        distance_from_goal = self.__calculate_distance_from_goal()
        angle_from_goal = self.__calculate_angle_from_goal()

        observation = list(self.__ranges[:])
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        if self.__collision_occurred():
            reward = self.__COLLISION_REWARD
            done = True
        elif distance_from_goal < self.__GOAL_DISTANCE_THRESHOLD:
            reward = self.__GOAL_REWARD
            done = True
        else:
            reward = (self.__TRANSITION_REWARD_FACTOR
                      * (self.__distance_from_goal - distance_from_goal))
            done = False

        self.__distance_from_goal = distance_from_goal

        return observation, reward, done, []

    def render(self, mode="human") -> None:
        plt.clf()
        walls = np.concatenate((self.__track, self.__obstacles_lines), axis=0)
        for wall in walls:
            plt.plot(wall[0], wall[1], 'b')

        for scan_line in self.__scan_lines:
            plt.plot(scan_line[0], scan_line[1], 'y')

        for scan_point in self.__scan_points:
            plt.plot(scan_point[0], scan_point[1], 'co')

        plt.plot(self.__pose[0], self.__pose[1], 'ro')
        plt.plot(self.__goal[0], self.__goal[1], 'go')
        plt.xlim((-12, 12))
        plt.ylim((-12, 12))

        plt.pause(0.01)

    def close(self) -> None:
        plt.close()
