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

WALL_DISTANCE_THRESHOLD = 0.4 # m
GOAL_DISTANCE_THRESHOLD = 0.4 # m

STEP_REWARD = -0.2
DIRECTION_REWARD = 1
TRANSITION_REWARD_FACTOR = 10

SCAN_RANGE_MAX = 30.0
SCAN_RANGE_MIN = 0.2
MAXIMUM_GOAL_DISTANCE = 70
FIRST_PERSPECTIVE_DIRECTION = 2
N_MEASUREMENTS = 5
N_OBSERVATIONS = N_MEASUREMENTS + 2

MAZE = (
    ((0, 0), (0, 20)),
    ((0, 20), (20, 20)),
    ((20, 20), (20, 0)),
    ((20, 0), (0, 0)),

    ((6, 6), (6, 7)),
    ((6, 7), (7, 7)),
    ((7, 7), (7, 6)),
    ((7, 6), (6, 6)),

    ((3, 3), (3, 4)),
    ((3, 4), (4, 4)),
    ((4, 4), (4, 3)),
    ((4, 3), (3, 3)),

    ((14, 14), (14,15)),
    ((14, 15), (15, 15)),
    ((15, 15), (15, 14)),
    ((15, 14), (14, 14)),

    ((16, 16), (16, 17)),
    ((16, 17), (17, 17)),
    ((17, 17), (17, 16)),
    ((17, 16), (16, 16))
    
)

SCAN_ANGLES = (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2)


class NavigationGoalEnv:

    def __init__(self, goal):
        self.ranges = np.empty((N_MEASUREMENTS,))
        self.pose = np.empty((3,))
        self.total_actions = 0
        self.goal = goal
        self.distance_from_goal = 0.0

        self.action_space = Discrete(N_ACTIONS)

        # ranges
        high = N_MEASUREMENTS * [SCAN_RANGE_MAX]
        # distance from goal
        high.append(MAXIMUM_GOAL_DISTANCE)
        # angle from goal
        high.append(math.pi)
        high = np.array(high, dtype=np.float32)

        # ranges
        low = N_MEASUREMENTS * [SCAN_RANGE_MIN]
        # distance from goal
        low.append(0)
        # angle from goal
        low.append(-math.pi)
        low = np.array(low, dtype=np.float32)

        self.observation_space = Box(
            low=low, high=high, shape=(N_OBSERVATIONS,), dtype=np.float32)

    def get_point(self, x0, y0, angle, d):
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
        self.scan_lines = []
        self.scan_points = []

        ranges = []
        angle_list = np.array(SCAN_ANGLES) + self.pose[2]
        x0 = self.pose[0]
        y0 = self.pose[1]

        for i in range(len(angle_list)):
            if angle_list[i] < -math.pi:
                angle_list[i] = 2 * math.pi + angle_list[i]
            elif angle_list[i] > math.pi:
                angle_list[i] = angle_list[i] - 2 * math.pi

        for angle in angle_list:
            x1, y1 = self.get_point(x0, y0, angle, SCAN_RANGE_MAX)
            scan_line = ((x0, x1), (y0, y1))
            self.scan_lines.append(scan_line) 

            min_dist = SCAN_RANGE_MAX
            min_x = x1
            min_y = y1
            for wall in MAZE:
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

            self.scan_points.append((min_x, min_y))
            sensor_noise = random.gauss(0, SENSOR_STANDARD_DEVIATION)
            min_dist += sensor_noise
            ranges.append(min_dist)

        self.ranges = np.array(ranges)

    def collision_occured(self):
        for range_ in self.ranges:
            if range_ < WALL_DISTANCE_THRESHOLD:
                return True

        return False

    def reset(self):
        plt.close()
        plt.figure(figsize=(19.20,10.80))
        self.total_actions = 0

        x = 10.0
        y = 10.0
        yaw = random.uniform(-math.pi, math.pi)
        self.pose = np.array([x, y, yaw])

        self.update_scan()

        uav_position = np.array([self.pose[0], self.pose[1]])
        distance_from_goal = np.linalg.norm(uav_position - self.goal)

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
        if not self.action_space.contains(action):
            print('Invalid action')

        self.perform_action(action)

        self.update_scan()

        uav_position = np.array([self.pose[0], self.pose[1]])
        distance_from_goal = np.linalg.norm(uav_position - self.goal)

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

        goal_reached = distance_from_goal < GOAL_DISTANCE_THRESHOLD

        if goal_reached:
            print('GOAL REACHED')

        done = goal_reached or self.collision_occured()

        environment_reward = np.sum(-100 * np.exp(-3 * np.array(self.ranges)))

        transition_reward = TRANSITION_REWARD_FACTOR * \
            (self.distance_from_goal - distance_from_goal)
        '''
        direction_reward = DIRECTION_REWARD * \
            int(np.argmax(self.ranges) == FIRST_PERSPECTIVE_DIRECTION)
        '''
        reward = STEP_REWARD + environment_reward + transition_reward
        '''
        print('rew {}, step {}, env {}, trans {}'.format(
            reward, STEP_REWARD, environment_reward, transition_reward))
        '''
        self.distance_from_goal = distance_from_goal

        return observation, reward, done, []

    def render(self):
        plt.clf()
        for point in MAZE:
            plt.plot(point[0], point[1], 'b')
        
        for scan_line in self.scan_lines:
            plt.plot(scan_line[0], scan_line[1], 'y')
        
        for scan_point in self.scan_points:
            plt.plot(scan_point[0], scan_point[1], 'co')

        plt.plot(self.pose[0], self.pose[1], 'ro')
        plt.plot(self.goal[0], self.goal[1], 'go')
        plt.xlim((-1, 21))
        plt.ylim((-1, 21))

        plt.pause(0.05)
    
    def close(self):
        pass

        
