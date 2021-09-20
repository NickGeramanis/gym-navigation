import math

from gym_navigation.utils.pose import Point


class TestPoint:
    def test_point_equality(self):
        point1 = Point(4, 2)
        point2 = Point(4, 2)

        assert point1 == point2

    def test_point_inequality(self):
        point1 = Point(4, 2)
        point2 = Point(8, 2)

        assert point1 != point2

    def test_calculate_distance(self):
        point1 = Point(4, 2)
        point2 = Point(8, 2)

        distance = point1.calculate_distance(point2)

        assert math.isclose(distance, 4)
