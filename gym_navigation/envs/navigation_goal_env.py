import numpy as np
import math
import random
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding

N_ACTIONS = 3
FORWARD = 0
YAW_RIGHT = 1
YAW_LEFT = 2

FORWARD_LINEAR_SHIFT = 0.2  # m
YAW_LINEAR_SHIFT = 0.04  # m
YAW_ANGULAR_SHIFT = 0.2  # rad

SHIFT_STANDARD_DEVIATION = 0.02
SENSOR_STANDARD_DEVIATION = 0.01

WALL_DISTANCE_THRESHOLD = 0.4  # m
GOAL_DISTANCE_THRESHOLD = 0.4  # m
MINIMUM_DISTANCE = 1  # m

TRANSITION_REWARD_FACTOR = 10
GOAL_REWARD = 200
COLLISION_REWARD = -200

#SCAN_ANGLES = (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2)
SCAN_ANGLES = (-math.pi/4, 0, math.pi/4)
#SCAN_ANGLES = (-math.pi/2, 0, math.pi/2)

SCAN_RANGE_MAX = 30.0
SCAN_RANGE_MIN = 0.2
N_MEASUREMENTS = len(SCAN_ANGLES)
MAXIMUM_GOAL_DISTANCE = 30
N_OBSERVATIONS = N_MEASUREMENTS + 2
N_OBSTACLES = 10
OBSTACLES_LENGTH = 0.5

TRACK = (
    ((-10, -10), (-10, 10)),
    ((-10, 10), (10, 10)),
    ((10, 10), (10, -10)),
    ((10, -10), (-10, -10))
)


class NavigationGoalEnv(gym.Env):

    def __init__(self):
        self.ranges = np.empty((N_MEASUREMENTS,))
        '''
        Pose = (x, y, yaw).
        Note that yaw is measured from the y axis and E [-pi, pi].
        '''
        self.pose = np.empty((3,))
        self.goal = np.empty((2,))
        self.obstacles = np.empty((N_OBSTACLES, 2))
        self.total_actions = 0
        self.distance_from_goal = 0.0

        self.scan_lines = np.empty((N_MEASUREMENTS,), dtype=object)
        self.scan_points = np.empty((N_MEASUREMENTS,), dtype=object)

        self.action_space = spaces.Discrete(N_ACTIONS)

        high = N_MEASUREMENTS * [SCAN_RANGE_MAX]
        high.append(MAXIMUM_GOAL_DISTANCE)
        high.append(math.pi)
        high = np.array(high, dtype=np.float32)

        low = N_MEASUREMENTS * [SCAN_RANGE_MIN]
        low.append(0)
        low.append(-math.pi)
        low = np.array(low, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def init_obstacles(self):
        for obstacle_i in range(N_OBSTACLES):
            while True:
                obstacle_x = random.uniform(-9, 9)
                obstacle_y = random.uniform(-9, 9)

                distance_from_pose = math.sqrt(
                    (obstacle_x - self.pose[0]) ** 2 + (obstacle_y - self.pose[1]) ** 2)

                distance_from_goal = math.sqrt(
                    (obstacle_x - self.goal[0]) ** 2 + (obstacle_y - self.goal[1]) ** 2)

                if distance_from_pose > MINIMUM_DISTANCE and distance_from_goal > MINIMUM_DISTANCE:
                    break

            self.obstacles[obstacle_i][0] = obstacle_x
            self.obstacles[obstacle_i][1] = obstacle_y

    def init_goal(self):
        while True:
            goal_x = random.uniform(-9, 9)
            goal_y = random.uniform(-9, 9)
            distance_from_pose = math.sqrt(
                (goal_x - self.pose[0]) ** 2 + (goal_x - self.pose[1]) ** 2)
            if distance_from_pose > MINIMUM_DISTANCE:
                break

        self.goal[0] = goal_x
        self.goal[1] = goal_y

    def init_pose(self):
        self.pose[0] = 0.0
        self.pose[1] = 0.0
        self.pose[2] = random.uniform(-math.pi, math.pi)

    def get_point(self, x0, y0, angle, d):
        '''
        Get the coordinates of a point that is at a distance d
        from the given position (x,y) and orientation (yaw).
        '''
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

    def perform_action(self, action):
        '''
        Change the pose of the robot depending on the given action.
        In each action we add white Gaussian noise: y = y_hat + w.
        '''
        linear_shift_noise = random.gauss(0, SHIFT_STANDARD_DEVIATION)
        angular_shift_noise = random.gauss(0, SHIFT_STANDARD_DEVIATION)

        if action == FORWARD:
            d = FORWARD_LINEAR_SHIFT + linear_shift_noise
            self.pose[2] += angular_shift_noise
        elif action == YAW_RIGHT:
            d = YAW_LINEAR_SHIFT + linear_shift_noise
            self.pose[2] += YAW_ANGULAR_SHIFT + angular_shift_noise
        else:
            d = YAW_LINEAR_SHIFT + linear_shift_noise
            self.pose[2] -= YAW_ANGULAR_SHIFT + angular_shift_noise

        # yaw must E [-pi,pi]
        if self.pose[2] < -math.pi:
            self.pose[2] = 2 * math.pi + self.pose[2]
        elif self.pose[2] > math.pi:
            self.pose[2] = self.pose[2] - 2 * math.pi

        self.pose[0], self.pose[1] = self.get_point(
            self.pose[0], self.pose[1], self.pose[2], d)

    def update_scan(self):
        # Get the range distance from each measurement angle.
        angle_list = np.array(SCAN_ANGLES) + self.pose[2]
        x0 = self.pose[0]
        y0 = self.pose[1]

        # Measurement angles must E [-pi,pi].
        for i in range(len(angle_list)):
            if angle_list[i] < -math.pi:
                angle_list[i] = 2 * math.pi + angle_list[i]
            elif angle_list[i] > math.pi:
                angle_list[i] = angle_list[i] - 2 * math.pi

        for i in range(N_MEASUREMENTS):
            x1, y1 = self.get_point(x0, y0, angle_list[i], SCAN_RANGE_MAX)
            scan_line = ((x0, x1), (y0, y1))
            self.scan_lines[i] = scan_line

            min_dist = SCAN_RANGE_MAX
            min_x = x1
            min_y = y1
            for wall in TRACK:
                x2 = wall[0][0]
                x3 = wall[0][1]
                y2 = wall[1][0]
                y3 = wall[1][1]

                denominator = (x0 - x1)*(y2 - y3) - (y0 - y1)*(x2 - x3)
                if denominator == 0:
                    continue

                nominator_x = (x0*y1 - y0*x1)*(x2 - x3) - \
                    (x0 - x1)*(x2*y3 - y2*x3)
                nominator_y = (x0*y1 - y0*x1)*(y2 - y3) - \
                    (y0 - y1)*(x2*y3 - y2*x3)

                x = nominator_x / denominator
                y = nominator_y / denominator

                not_in_scan_line = (x1 > x0 and (x < x0 or x > x1)) or (x0 > x1 and (x < x1 or x > x0)) or (
                    y1 > y0 and (y < y0 or y > y1)) or (y0 > y1 and (y < y1 or y > y0))
                not_in_wall_line = (x3 > x2 and (x < x2 or x > x3)) or (x2 > x3 and (x < x3 or x > x2)) or (
                    y3 > y2 and (y < y2 or y > y3)) or (y2 > y3 and (y < y3 or y > y2))
                if not_in_scan_line or not_in_wall_line:
                    continue

                dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_x = x
                    min_y = y

            self.scan_points[i] = (min_x, min_y)
            sensor_noise = random.gauss(0, SENSOR_STANDARD_DEVIATION)
            min_dist += sensor_noise
            self.ranges[i] = min_dist

    def collision_occurred(self):
        for range_ in self.ranges:
            if range_ < WALL_DISTANCE_THRESHOLD:
                return True

        return False

    def reset(self):
        plt.close()
        plt.figure(figsize=(6.40, 4.80))
        self.total_actions = 0

        self.init_pose()
        self.init_goal()
        self.init_obstacles()

        self.update_scan()

        position = np.array([self.pose[0], self.pose[1]])
        distance_from_goal = np.linalg.norm(position - self.goal)

        goal_angle = math.atan2(
            self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        angle_from_goal = goal_angle - self.pose[2]
        if angle_from_goal > math.pi:
            angle_from_goal -= 2 * math.pi
        elif angle_from_goal < -math.pi:
            angle_from_goal += 2 * math.pi

        observation = list(self.ranges[:])
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        self.distance_from_goal = distance_from_goal

        return observation

    def step(self, action):
        assert self.action_space.contains(
            action), '{} ({}) invalid'.format(action, type(action))

        self.perform_action(action)

        self.update_scan()

        position = np.array([self.pose[0], self.pose[1]])
        distance_from_goal = np.linalg.norm(position - self.goal)

        goal_angle = math.atan2(
            self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        angle_from_goal = goal_angle - self.pose[2]
        if angle_from_goal > math.pi:
            angle_from_goal -= 2 * math.pi
        elif angle_from_goal < -math.pi:
            angle_from_goal += 2 * math.pi

        observation = list(self.ranges[:])
        observation.append(distance_from_goal)
        observation.append(angle_from_goal)

        if self.collision_occurred():
            reward = COLLISION_REWARD
            done = True
        elif distance_from_goal < GOAL_DISTANCE_THRESHOLD:
            reward = GOAL_REWARD
            done = True
        else:
            reward = TRANSITION_REWARD_FACTOR * \
                (self.distance_from_goal - distance_from_goal)
            done = False

        self.distance_from_goal = distance_from_goal

        return observation, reward, done, []

    def render(self):
        plt.clf()
        for point in TRACK:
            plt.plot(point[0], point[1], 'b')

        for scan_line in self.scan_lines:
            plt.plot(scan_line[0], scan_line[1], 'y')

        for scan_point in self.scan_points:
            plt.plot(scan_point[0], scan_point[1], 'co')

        plt.plot(self.pose[0], self.pose[1], 'ro')
        plt.plot(self.goal[0], self.goal[1], 'go')
        plt.xlim((-12, 12))
        plt.ylim((-12, 12))

        plt.pause(0.05)

    def close(self):
        plt.close()
