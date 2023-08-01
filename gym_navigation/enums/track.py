"""This module contains the Track enum."""
from enum import Enum
from typing import Tuple

from gym_navigation.geometry.line import Line
from gym_navigation.geometry.point import Point


class Track(Enum):
    """The Track enum."""

    walls: Tuple[Line, ...]
    spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]

    def __new__(cls,
                value: int,
                walls: Tuple[Line, ...] = (),
                spawn_area: Tuple[Tuple[Tuple[float, float],
                                        Tuple[float, float]], ...] = ()):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.walls = walls
        obj.spawn_area = spawn_area
        return obj

    TRACK1 = (1,
              (
                  Line(Point(0, 0), Point(0, 20)),
                  Line(Point(0, 20), Point(20, 20)),
                  Line(Point(20, 20), Point(20, 8.5)),
                  Line(Point(20, 8.5), Point(11.5, 8.5)),
                  Line(Point(11.5, 8.5), Point(11.5, 0)),
                  Line(Point(11.5, 0), Point(0, 0)),
                  Line(Point(3, 3), Point(3, 17)),
                  Line(Point(3, 17), Point(17, 17)),
                  Line(Point(17, 17), Point(17, 11.5)),
                  Line(Point(17, 11.5), Point(8.5, 11.5)),
                  Line(Point(8.5, 11.5), Point(8.5, 3)),
                  Line(Point(8.5, 3), Point(3, 3))
              ),
              (
                  ((1.5, 1.5), (1.5, 18.5)),
                  ((1.5, 18.5), (18.5, 18.5)),
                  ((18.5, 18.5), (10, 18.5)),
                  ((10, 18.5), (10, 10)),
                  ((10, 10), (1.5, 10)),
                  ((1.5, 10), (1.5, 1.5))
              )
              )

    TRACK2 = (2,
              (
                  Line(Point(0, 0), Point(0, 20)),
                  Line(Point(0, 20), Point(20, 20)),
                  Line(Point(20, 20), Point(20, 0)),
                  Line(Point(20, 0), Point(0, 0))
              ),
              (
                  ((1, 19), (1, 19)),
              )
              )
