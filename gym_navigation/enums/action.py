"""This module contains the Action enum."""
from enum import Enum


class Action(Enum):
    """The Action enum."""

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, linear_shift: float, angular_shift: float) -> None:
        self.linear_shift = linear_shift
        self.angular_shift = angular_shift

    FORWARD = 0.2, 0
    ROTATE_RIGHT = 0.04, 0.2
    ROTATE_LEFT = 0.04, -0.2
