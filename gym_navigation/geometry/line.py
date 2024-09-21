"""This module contains the Line class."""
from __future__ import annotations

import math
from dataclasses import dataclass

from gym_navigation.geometry.point import Point


class NoIntersectionError(Exception):
    """Exception when there is no intersection between two lines."""


@dataclass
class Line:
    """A line (line segment) in Cartesian plane."""

    start: Point
    end: Point
    slope: float
    y_intercept: float

    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        if start == end:
            raise RuntimeError('Equal start and end points of the line')
        if start.x_coordinate == end.x_coordinate:  # Vertical line
            self.slope = 0
            self.y_intercept = math.inf
        elif start.y_coordinate == end.y_coordinate:  # Horizontal line
            self.slope = 0
            self.y_intercept = start.y_coordinate
        else:
            self.slope = ((start.y_coordinate - end.y_coordinate)
                          / (start.x_coordinate - end.x_coordinate))
            self.y_intercept = (
                start.y_coordinate - self.slope * start.x_coordinate)

    def get_intersection(self, other: Line) -> Point:
        """Get the intersection point between two lines.

        Raise an error if it does not exist.
        """
        if self.slope == other.slope:
            raise NoIntersectionError('Parallel lines')

        if self.start.x_coordinate == self.end.x_coordinate:
            x_coordinate = self.start.x_coordinate
            y_coordinate = other.slope * x_coordinate + other.y_intercept
        elif other.start.x_coordinate == other.end.x_coordinate:
            x_coordinate = other.start.x_coordinate
            y_coordinate = self.slope * x_coordinate + self.y_intercept
        elif self.start.y_coordinate == self.end.y_coordinate:
            y_coordinate = self.start.y_coordinate
            x_coordinate = (y_coordinate - other.y_intercept) / other.slope
        elif other.start.y_coordinate == other.end.y_coordinate:
            y_coordinate = other.start.y_coordinate
            x_coordinate = (y_coordinate - self.y_intercept) / self.slope
        else:
            x_coordinate = ((self.y_intercept - other.y_intercept)
                            / (other.slope - self.slope))
            y_coordinate = self.slope * x_coordinate + self.y_intercept

        intersection = Point(x_coordinate, y_coordinate)

        if self.contains(intersection) and other.contains(intersection):
            return intersection

        raise NoIntersectionError('No intersection')

    def contains(self, point: Point) -> bool:
        """Calculate if the line contains a given point."""
        contains_x = (
            min(self.start.x_coordinate, self.end.x_coordinate)
            <= point.x_coordinate
            <= max(self.start.x_coordinate, self.end.x_coordinate))
        contains_y = (
            min(self.start.y_coordinate, self.end.y_coordinate)
            <= point.y_coordinate
            <= max(self.start.y_coordinate, self.end.y_coordinate))
        return contains_x and contains_y
