"""This module contains the Line class."""
import math
from typing import Any

from gym_navigation.utils.point import Point


class NoIntersection(Exception):
    """Exception when there is no intersection between two lines."""


class Line:
    """A line (line segment) in Cartesian plane."""

    __start: Point
    __end: Point
    __slope: float
    __y_intercept: float

    def __init__(self, start: Point, end: Point) -> None:
        self.__start = start
        self.__end = end
        if start.x == end.x:  # Vertical line
            self.__slope = 0
            self.__y_intercept = math.inf
        elif start.y == end.y:  # Horizontal line
            self.__slope = 0
            self.__y_intercept = start.y
        else:
            self.__slope = (start.y - end.y) / (start.x - end.x)
            self.__y_intercept = start.y - self.__slope * start.x

    def get_intersection(self, other) -> Point:
        """Get the intersection point between two lines.

        Raise an error if it does not exist.
        """
        if self.__slope == other.slope:  # Parallel lines
            raise NoIntersection

        if self.__start.x == self.__end.x:
            x = self.__start.x
            y = other.slope * x + other.y_intercept
        elif other.start.x == other.end.x:
            x = other.start.x
            y = self.__slope * x + self.__y_intercept
        elif self.__start.y == self.__end.y:
            y = self.__start.y
            x = (y - other.y_intercept) / other.slope
        elif other.start.y == other.end.y:
            y = other.start.y
            x = (y - self.__y_intercept) / self.__slope
        else:
            x = ((self.__y_intercept - other.y_intercept)
                 / (other.slope - self.__slope))
            y = self.__slope * x + self.__y_intercept

        intersection = Point(x, y)

        if self.contains(intersection) and other.contains(intersection):
            return intersection

        raise NoIntersection

    def contains(self, point: Point) -> bool:
        """Calculate if the line contains a given point."""
        contains_x = (min(self.__start.x, self.__end.x) <= point.x
                      <= max(self.__start.x, self.__end.x))
        contains_y = (min(self.__start.y, self.__end.y) <= point.y
                      <= max(self.__start.y, self.__end.y))
        return contains_x and contains_y

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Line)
                and (self.__start == other.start and self.__end == other.end
                or self.__start == other.end and self.__end == other.start))

    @property
    def start(self) -> Point:
        """The start point of the line."""
        return self.__start

    @start.setter
    def start(self, start) -> None:
        self.__start = start

    @property
    def end(self) -> Point:
        """The end point of the line."""
        return self.__end

    @end.setter
    def end(self, end) -> None:
        self.__end = end

    @property
    def slope(self) -> float:
        """The slope of the line equation."""
        return self.__slope

    @slope.setter
    def slope(self, slope) -> None:
        self.__slope = slope

    @property
    def y_intercept(self) -> float:
        """The y-intercept of the line equation."""
        return self.__y_intercept

    @y_intercept.setter
    def y_intercept(self, y_intercept) -> None:
        self.__y_intercept = y_intercept
