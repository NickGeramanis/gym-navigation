import math

from gym_navigation.geometry.point import Point


def test_point_equality():
    point1 = Point(4, 2)
    point2 = Point(4, 2)

    assert point1 == point2


def test_point_inequality():
    point1 = Point(4, 2)
    point2 = Point(8, 2)

    assert point1 != point2


def test_calculate_distance():
    point1 = Point(4, 2)
    point2 = Point(8, 2)

    distance = point1.calculate_distance(point2)

    assert math.isclose(distance, 4)
