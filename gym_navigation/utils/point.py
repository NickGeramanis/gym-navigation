from math import sqrt, isclose


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def calculate_distance(self, other) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other) -> bool:
        return isclose(self.x, other.x) and isclose(self.y, other.y)

    def __str__(self) -> str:
        return f'Point({self.x}, {self.y})'
