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

WALL_DISTANCE_THRESHOLD = 0.4

MAX_ACTIONS = 500

COLLISION_REWARD = -200
FORWARD_REWARD = +5
YAW_REWARD = -0.5
STEP_REWARD = 0

#SCAN_ANGLES = (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2)
SCAN_ANGLES = (-math.pi/4, 0, math.pi/4)
#SCAN_ANGLES = (-math.pi/2, 0, math.pi/2)

SCAN_RANGE_MAX = 30.0
SCAN_RANGE_MIN = 0.2
N_MEASUREMENTS = len(SCAN_ANGLES)

TRACK = (
    ((-10, -10), (-10, 10)),
    ((-10, 10), (10, 10)),
    ((10, 10), (10, -1.5)),
    ((10, 1.5), (-1.5, -1.5)),
    ((1.5, 1.5), (-1.5, -10)),
    ((1.5, -10), (-10, -10)),

    ((-7, -7), (-7, 7)),
    ((-7, 7), (7, 7)),
    ((7, 7), (7, 1.5)),
    ((7, -1.5), (1.5, 1.5)),
    ((-1.5, -1.5), (1.5, -7)),
    ((-1.5, -7), (-7, -7))
)

SPAWNABLE_AREA = (
    ((-8.5, -8.5), (-8.5, 8.5)),
    ((-8.5, 8.5), (8.5, 8.5)),
    ((8.5, 8.5), (0, 8.5)),
    ((0, 8.5), (0, 0)),
    ((0, 0), (-8.5, 0)),
    ((-8.5, 0), (-8.5, -8.5))
)


class NavigationEnv(gym.Env):

    def __init__(self):
        self.ranges = np.empty((N_MEASUREMENTS,))
        '''
        Pose = (x, y, yaw).
        Note that yaw is measured from the y axis and E [-pi, pi].
        '''
        self.pose = np.empty((3,))
        self.total_actions = 0

        self.scan_lines = np.empty((N_MEASUREMENTS,), dtype=object)
        self.scan_points = np.empty((N_MEASUREMENTS,), dtype=object)

        self.action_space = spaces.Discrete(N_ACTIONS)

        self.observation_space = spaces.Box(
            low=SCAN_RANGE_MAX, high=SCAN_RANGE_MIN, shape=(N_MEASUREMENTS,), dtype=np.float32)

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

        # Yaw must E [-pi,pi].
        if self.pose[2] < -math.pi:
            self.pose[2] = 2 * math.pi + self.pose[2]
        elif self.pose[2] > math.pi:
            self.pose[2] = self.pose[2] - 2 * math.pi

        self.pose[0], self.pose[1] = self.get_point(
            self.pose[0], self.pose[1], self.pose[2], d)

    def update_scan(self):
        '''
        Get the range distance from each measurement angle.
        '''
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

    def collision_occured(self):
        for range_ in self.ranges:
            if range_ < WALL_DISTANCE_THRESHOLD:
                return True

        return False

    def reset(self):
        plt.close()
        plt.figure(figsize=(6.40, 4.80))

        self.total_actions = 0

        # Random initial pose.
        area = random.choice(SPAWNABLE_AREA)
        self.pose[0] = random.uniform(area[0][0], area[0][1])
        self.pose[1] = random.uniform(area[1][0], area[1][1])
        self.pose[2] = random.uniform(-math.pi, math.pi)

        self.update_scan()
        observation = list(self.ranges)

        return observation

    def step(self, action):
        assert self.action_space.contains(
            action), '{} ({}) invalid'.format(action, type(action))

        self.total_actions += 1

        self.perform_action(action)

        self.update_scan()
        observation = list(self.ranges)

        collision_occured = self.collision_occured()

        '''
        Have a maximum number of action to avoid infinite long episodes.
        '''
        done = True if collision_occured or self.total_actions == MAX_ACTIONS else False

        if collision_occured:
            reward = COLLISION_REWARD
        elif action == FORWARD:
            reward = FORWARD_REWARD
        else:
            reward = YAW_REWARD

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

        plt.xlim((-12, 12))
        plt.ylim((-12, 12))

        plt.pause(0.01)

    def close(self):
        plt.close()
