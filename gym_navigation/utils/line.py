import math

from gym_navigation.utils.point import Point


class NoIntersection(Exception):
    """An exception that is used when there is no intersection
    between two lines.
    """
    pass


class Line:
    """A line (line segment) in Cartesian plane."""
    __start: Point
    __end: Point
    __m: float
    __b: float

    def __init__(self, start: Point, end: Point) -> None:
        self.__start = start
        self.__end = end
        if start.x == end.x:  # Vertical line
            self.__m = 0
            self.__b = math.inf
        elif start.y == end.y:  # Horizontal line
            self.__m = 0
            self.__b = start.y
        else:
            self.__m = (start.y - end.y) / (start.x - end.x)
            self.__b = start.y - self.__m * start.x

    def get_intersection(self, other) -> Point:
        """Get the intersection point between two lines.
        Raise an error if it does not exist.
        """
        if self.__m == other.m:  # Parallel lines
            raise NoIntersection

        if self.__start.x == self.__end.x:
            x = self.__start.x
            y = other.m * x + other.b
        elif other.start.x == other.end.x:
            x = other.start.x
            y = self.__m * x + self.__b
        elif self.__start.y == self.__end.y:
            y = self.__start.y
            x = (y - other.b) / other.m
        elif other.start.y == other.end.y:
            y = other.start.y
            x = (y - self.__b) / self.__m
        else:
            x = (self.__b - other.b) / (other.m - self.__m)
            y = self.__m * x + self.__b

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

    def __eq__(self, other) -> bool:
        return (self.__start == other.start and self.__end == other.end
                or self.__start == other.end and self.__end == other.start)

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
    def m(self) -> float:
        """The start point of the line."""
        return self.__m

    @m.setter
    def m(self, m) -> None:
        """The m (gradient) coefficient of the line equation."""
        self.__m = m

    @property
    def b(self) -> float:
        return self.__b

    @b.setter
    def b(self, b) -> None:
        """The b coefficient of the line equation."""
        self.__b = b
