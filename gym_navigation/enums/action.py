"""This module contains the Action enum."""
from enum import Enum


class Action(Enum):
    """The Action enum."""

    linear_shift: float
    angular_shift: float

    def __new__(cls,
                value: int,
                linear_shift: float = 0,
                angular_shift: float = 0):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.linear_shift = linear_shift
        obj.angular_shift = angular_shift
        return obj

    FORWARD = (0, 0.2, 0)
    ROTATE_RIGHT = (1, 0.04, 0.2)
    ROTATE_LEFT = (2, 0.04, -0.2)
