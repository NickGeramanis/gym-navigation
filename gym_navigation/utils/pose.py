from math import tan, pi, sqrt, atan2, isclose

from gym_navigation.utils.point import Point


class Pose:
    def __init__(self, position: Point, yaw: float) -> None:
        self.position = position
        self.yaw = yaw
        # yaw is measured from the y axis and E [-pi, pi]
        self.__correct_yaw()

    def move(self, d: float) -> None:
        if self.yaw == 0:
            self.position.y += d
        elif abs(self.yaw) == pi:
            self.position.y -= d
        else:
            m = tan(pi / 2 - self.yaw)
            starting_x = self.position.x
            if self.yaw < 0:
                self.position.x -= sqrt(d ** 2 / (m ** 2 + 1))
            else:
                self.position.x += sqrt(d ** 2 / (m ** 2 + 1))
            self.position.y -= m * (starting_x - self.position.x)

    def rotate(self, theta: float) -> None:
        self.yaw += theta
        self.__correct_yaw()

    def shift(self, d: float, theta: float) -> None:
        # Do we move first and then rotate or the other way around?
        self.move(d)
        self.rotate(theta)

    def calculate_angle_difference(self, other: Point) -> float:
        vector1 = Point(other.x - self.position.x, other.y - self.position.y)

        pose2 = Pose(Point(self.position.x, self.position.y), self.yaw)
        pose2.move(1)
        vector2 = Point(pose2.position.x - self.position.x,
                        pose2.position.y - self.position.y)

        angle_difference = atan2(vector1.x * vector2.y - vector1.y * vector2.x,
                                 vector1.x * vector2.x + vector1.y * vector2.y)
        return angle_difference

    def __correct_yaw(self) -> None:
        if self.yaw < -pi:
            self.yaw += 2 * pi
        elif self.yaw > pi:
            self.yaw -= 2 * pi

    def __eq__(self, other) -> bool:
        return (self.position == other.position
                and isclose(self.yaw, other.yaw))
