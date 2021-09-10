from typing import Optional

from gym_navigation.utils.point import Point


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        if self.start.x == self.end.x:  # Vertical line
            self.m = 0
            self.b = None
        elif self.start.y == self.end.y:  # Horizontal line
            self.m = 0
            self.b = self.start.y
        else:
            self.m = (self.start.y - self.end.y) / (self.start.x - self.end.x)
            self.b = self.start.y - self.m * self.start.x

    def get_intersection(self, other) -> Optional[Point]:
        if self.m == other.m:  # Parallel lines
            return None
        elif self.start.x == self.end.x:
            x = self.start.x
            y = other.m * x + other.b
        elif other.start.x == other.end.x:
            x = other.start.x
            y = self.m * x + self.b
        elif self.start.y == self.end.y:
            y = self.start.y
            x = (y - other.b) / other.m
        elif other.start.y == other.end.y:
            y = other.start.y
            x = (y - self.b) / self.m
        else:
            x = (self.b - other.b) / (other.m - self.m)
            y = self.m * x + self.b

        intersection = Point(x, y)

        if self.contains(intersection) and other.contains(intersection):
            return intersection

        return None

    def contains(self, point: Point) -> bool:
        contains_x = (min(self.start.x, self.end.x) <= point.x
                      <= max(self.start.x, self.end.x))
        contains_y = (min(self.start.y, self.end.y) <= point.y
                      <= max(self.start.y, self.end.y))
        return contains_x and contains_y

    def __eq__(self, other) -> bool:
        return (self.start == other.start and self.end == other.end or
                self.start == other.end and self.end == other.start)
