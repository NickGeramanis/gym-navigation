"""This module contains the Point class."""
from __future__ import annotations

import math
from typing import Any


class Point:
    """A point in Cartesian plane."""

    x_coordinate: float
    y_coordinate: float

    def __init__(self, x_coordinate: float, y_coordinate: float) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate

    def calculate_distance(self, other: Point) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((self.x_coordinate - other.x_coordinate) ** 2
                         + (self.y_coordinate - other.y_coordinate) ** 2)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Point)
                and math.isclose(self.x_coordinate, other.x_coordinate)
                and math.isclose(self.y_coordinate, other.y_coordinate))
