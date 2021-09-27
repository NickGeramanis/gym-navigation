import math

from gym_navigation.utils.point import Point
from typing import Any


class Pose:
    """The pose of an object in Cartesian plane."""

    __position: Point
    __yaw: float

    def __init__(self, position: Point, yaw: float) -> None:
        self.__position = position
        self.__yaw = yaw
        self.__correct_yaw()

    def move(self, d: float) -> None:
        """Move the pose by d."""
        if self.__yaw == 0:
            self.__position.y += d
        elif abs(self.__yaw) == math.pi:
            self.__position.y -= d
        else:
            m = math.tan(math.pi / 2 - self.__yaw)
            starting_x = self.__position.x
            if self.__yaw < 0:
                self.__position.x -= math.sqrt(
                    d ** 2 / (m ** 2 + 1))
            else:
                self.__position.x += math.sqrt(d ** 2 / (m ** 2 + 1))
            self.__position.y -= m * (starting_x - self.__position.x)

    def rotate(self, theta: float) -> None:
        """Rotate the yaw of the object by theta."""
        self.__yaw += theta
        self.__correct_yaw()

    def shift(self, d: float, theta: float) -> None:
        """Execute a shift.

        A shift is a movement followed by a rotation.
        In this implementation we move first and then rotate.
        """
        self.move(d)
        self.rotate(theta)

    def calculate_angle_difference(self, target: Point) -> float:
        """Calculate the angle difference from a point.

        This is the angle and the direction (+ or -) that the
        object need to rotate in order to face the target point.
        """
        vector1 = Point(target.x - self.__position.x,
                        target.y - self.__position.y)

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

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Pose)
                and self.__position == other.position
                and math.isclose(self.__yaw, other.yaw))

    @property
    def position(self) -> Point:
        """The position of the object in the Cartesian plane."""
        return self.__position

    @position.setter
    def position(self, position) -> None:
        self.__position = position

    @property
    def yaw(self) -> float:
        """The rotation (yaw) of the object.

        It is messearued from the y axis and E [-pi, pi].
        Positive yaw means clockwise direction while
        negative yaw means counterclockwise direction.
        """
        return self.__yaw

    @yaw.setter
    def yaw(self, yaw) -> None:
        self.__yaw = yaw
        self.__correct_yaw()
