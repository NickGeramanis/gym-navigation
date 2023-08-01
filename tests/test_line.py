import math

import pytest

from gym_navigation.geometry.line import Line, NoIntersectionError
from gym_navigation.geometry.point import Point


def test_line_equality():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(5, 0), Point(0, 10))

    assert line1 == line2


def test_line_inequality():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(0, -2), Point(2, 0))

    assert line1 != line2


def test_intersection_exists():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(2, 0), Point(7, 5))

    intersection = line1.get_intersection(line2)

    assert intersection == Point(4, 2)


def test_intersection_not_in_line():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(0, -2), Point(2, 0))

    with pytest.raises(NoIntersectionError):
        _ = line1.get_intersection(line2)


def test_intersection_parallel_lines():
    line1 = Line(Point(2, 0), Point(7, 5))
    line2 = Line(Point(1, 4), Point(2, 5))

    with pytest.raises(NoIntersectionError):
        _ = line1.get_intersection(line2)


def test_intersection_horizontal_line():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(0, 2), Point(10, 2))

    intersection1 = line1.get_intersection(line2)
    intersection2 = line2.get_intersection(line1)

    assert intersection1 == intersection2 == Point(4, 2)


def test_intersection_vertical_line():
    line1 = Line(Point(5, 0), Point(0, 10))
    line2 = Line(Point(2, 0), Point(2, 10))

    intersection1 = line1.get_intersection(line2)
    intersection2 = line2.get_intersection(line1)

    assert intersection1 == intersection2 == Point(2, 6)


def test_contains():
    line = Line(Point(2, 0), Point(7, 5))

    assert line.contains(Point(4, 2))


def test_not_contains():
    line = Line(Point(2, 0), Point(7, 5))

    assert not line.contains(Point(0, -2))


def test_correct_slope_and_y_intercept():
    line = Line(Point(2, 0), Point(7, 5))

    assert line.slope == 1 and line.y_intercept == -2


def test_correct_slope_and_y_intercept_vertical_line():
    line = Line(Point(2, 0), Point(2, 10))

    assert line.slope == 0 and line.y_intercept is math.inf


def test_correct_slope_and_y_intercept_horizontal_line():
    line = Line(Point(0, 2), Point(10, 2))

    assert line.slope == 0 and line.y_intercept == 2


def test_same_start_end_points():
    with pytest.raises(RuntimeError):
        _ = Line(Point(0, 0), Point(0, 0))
