import math

from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose


def test_pose_equality():
    pose1 = Pose(Point(1, 2), math.pi / 4)
    pose2 = Pose(Point(1, 2), math.pi / 4)

    assert pose1 == pose2


def test_pose_inequality():
    pose1 = Pose(Point(1, 2), math.pi / 4)
    pose2 = Pose(Point(1, 2), math.pi / 3)

    assert pose1 != pose2


def test_valid_yaw_greater_than_pi():
    pose = Pose(Point(1, 2), 3 * math.pi / 2)

    assert pose.yaw == -math.pi / 2


def test_valid_yaw_less_than_negative_pi():
    pose = Pose(Point(1, 2), -3 * math.pi / 2)

    assert pose.yaw == math.pi / 2


def test_move_with_yaw_zero():
    pose = Pose(Point(1, 2), 0)

    pose.move(1)

    assert pose == Pose(Point(1, 3), 0)


def test_move_with_yaw_pi():
    pose = Pose(Point(1, 2), math.pi)

    pose.move(1)

    assert pose == Pose(Point(1, 1), math.pi)


def test_move_with_positive_yaw():
    pose = Pose(Point(1, 2), math.pi / 4)

    pose.move(math.sqrt(2))

    assert pose == Pose(Point(2, 3), math.pi / 4)


def test_move_with_negative_yaw():
    pose = Pose(Point(1, 2), -math.pi / 4)

    pose.move(math.sqrt(2))

    assert pose == Pose(Point(0, 3), -math.pi / 4)


def test_shift():
    pose = Pose(Point(1, 2), math.pi / 4)

    pose.shift(math.sqrt(2), math.pi / 4)

    assert pose == Pose(Point(2, 3), math.pi / 2)


def test_rotate():
    pose = Pose(Point(1, 2), math.pi / 4)

    pose.rotate(math.pi / 4)

    assert pose == Pose(Point(1, 2), math.pi / 2)


def test_calculate_angle_difference_positive():
    pose = Pose(Point(1, 2), math.pi / 4)

    goal = Point(2, 2)
    angle_difference = pose.calculate_angle_difference(goal)

    assert math.isclose(angle_difference, math.pi / 4)


def test_calculate_angle_difference_negative():
    pose = Pose(Point(1, 2), math.pi / 4)

    goal = Point(1, 3)
    angle_difference = pose.calculate_angle_difference(goal)

    assert math.isclose(angle_difference, -math.pi / 4)
