"""This module contains the Color enum."""
from enum import Enum


class Color(Enum):
    """The Color enum."""

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
