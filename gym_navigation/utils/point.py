import math


class Point:
    """A point in Cartesian plane."""

    __x: float
    __y: float

    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y

    def calculate_distance(self, other) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((self.__x - other.x) ** 2 + (self.__y - other.y) ** 2)

    def __eq__(self, other) -> bool:
        return (math.isclose(self.__x, other.x)
                and math.isclose(self.__y, other.y))

    @property
    def x(self) -> float:
        """The x coordinate."""
        return self.__x

    @x.setter
    def x(self, x) -> None:
        self.__x = x

    @property
    def y(self) -> float:
        """The y coordinate."""
        return self.__y

    @y.setter
    def y(self, y) -> None:
        self.__y = y
