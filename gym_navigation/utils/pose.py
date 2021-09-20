import math

from gym_navigation.utils.point import Point


class Pose:
    __position: Point
    __yaw: float

    def __init__(self, position: Point, yaw: float) -> None:
        self.__position = position
        self.__yaw = yaw
        # yaw is measured from the y axis and E [-pi, pi]
        self.__correct_yaw()

    def move(self, d: float) -> None:
        if self.__yaw == 0:
            self.__position.y += d
        elif abs(self.__yaw) == math.pi:
            self.__position.y -= d
        else:
            m = math.tan(math.pi / 2 - self.__yaw)
            starting_x = self.__position.x
            if self.__yaw < 0:
                self.__position.x -= math.sqrt(d ** 2 / (m ** 2 + 1))
            else:
                self.__position.x += math.sqrt(d ** 2 / (m ** 2 + 1))
            self.__position.y -= m * (starting_x - self.__position.x)

    def rotate(self, theta: float) -> None:
        self.__yaw += theta
        self.__correct_yaw()

    def shift(self, d: float, theta: float) -> None:
        # Do we move first and then rotate or the other way around?
        self.move(d)
        self.rotate(theta)

    def calculate_angle_difference(self, other: Point) -> float:
        vector1 = Point(other.x - self.__position.x,
                        other.y - self.__position.y)

        pose2 = Pose(Point(self.__position.x, self.__position.y), self.__yaw)
        pose2.move(1)
        vector2 = Point(pose2.__position.x - self.__position.x,
                        pose2.__position.y - self.__position.y)

        angle_difference = math.atan2(
            vector1.x * vector2.y - vector1.y * vector2.x,
            vector1.x * vector2.x + vector1.y * vector2.y)
        return angle_difference

    def __correct_yaw(self) -> None:
        if self.__yaw < -math.pi:
            self.__yaw += 2 * math.pi
        elif self.__yaw > math.pi:
            self.__yaw -= 2 * math.pi

    def __eq__(self, other) -> bool:
        return (self.__position == other.position
                and math.isclose(self.__yaw, other.yaw))

    @property
    def position(self) -> Point:
        return self.__position

    @position.setter
    def position(self, position) -> None:
        self.__position = position

    @property
    def yaw(self) -> float:
        return self.__yaw

    @yaw.setter
    def yaw(self, yaw) -> None:
        self.__yaw = yaw
