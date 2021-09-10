from gym_navigation.utils.line import Line
from gym_navigation.utils.point import Point


class TestLine:
    def test_line_equality(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(5, 0), Point(0, 10))
        assert line1 == line2

    def test_line_inequality(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(0, -2), Point(2, 0))
        assert line1 != line2

    def test_intersection_exists(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(2, 0), Point(7, 5))
        intersection = line1.get_intersection(line2)
        assert intersection == Point(4, 2)

    def test_intersection_not_in_line(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(0, -2), Point(2, 0))
        intersection = line1.get_intersection(line2)
        assert intersection is None

    def test_intersection_parallel_lines(self):
        line1 = Line(Point(2, 0), Point(7, 5))
        line2 = Line(Point(1, 4), Point(2, 5))
        intersection = line1.get_intersection(line2)
        assert intersection is None

    def test_intersection_horizontal_line(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(0, 2), Point(10, 2))
        intersection1 = line1.get_intersection(line2)
        intersection2 = line2.get_intersection(line1)
        assert intersection1 == intersection2 == Point(4, 2)

    def test_intersection_vertical_line(self):
        line1 = Line(Point(5, 0), Point(0, 10))
        line2 = Line(Point(2, 0), Point(2, 10))
        intersection1 = line1.get_intersection(line2)
        intersection2 = line2.get_intersection(line1)
        assert intersection1 == intersection2 == Point(2, 6)

    def test_contains(self):
        line = Line(Point(2, 0), Point(7, 5))
        assert line.contains(Point(4, 2))

    def test_not_contains(self):
        line = Line(Point(2, 0), Point(7, 5))
        assert not line.contains(Point(0, -2))

    def test_correct_m_and_b(self):
        line = Line(Point(2, 0), Point(7, 5))
        assert line.m == 1 and line.b == -2

    def test_correct_m_and_b_vertical_line(self):
        line = Line(Point(2, 0), Point(2, 10))
        assert line.m == 0 and line.b is None

    def test_correct_m_and_b_horizontal_line(self):
        line = Line(Point(0, 2), Point(10, 2))
        assert line.m == 0 and line.b == 2

