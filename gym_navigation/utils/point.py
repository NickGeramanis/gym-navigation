import math


class Point:
    __x: float
    __y: float

    def __init__(self, x: float, y: float) -> None:
        self.__x = x
        self.__y = y

    def calculate_distance(self, other) -> float:
        return math.sqrt((self.__x - other.x) ** 2 + (self.__y - other.y) ** 2)

    def __eq__(self, other) -> bool:
        return (math.isclose(self.__x, other.x)
                and math.isclose(self.__y, other.y))

    @property
    def x(self) -> float:
        return self.__x

    @x.setter
    def x(self, x) -> None:
        self.__x = x

    @property
    def y(self) -> float:
        return self.__y

    @y.setter
    def y(self, y) -> None:
        self.__y = y
