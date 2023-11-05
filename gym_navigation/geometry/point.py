"""This module contains the Point class."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Point:
    """A point in Cartesian plane."""

    x_coordinate: float
    y_coordinate: float

    def calculate_distance(self, other: Point) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((self.x_coordinate - other.x_coordinate) ** 2
                         + (self.y_coordinate - other.y_coordinate) ** 2)
