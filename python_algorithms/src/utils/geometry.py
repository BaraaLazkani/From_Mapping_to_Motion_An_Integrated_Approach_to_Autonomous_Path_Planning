"""
Geometric utility functions for path planning.

This module provides fundamental geometric operations used across
multiple path planning algorithms, including line segment intersection
testing and collision detection.
"""

from typing import Tuple


def ccw(A: Tuple[float, float],
        B: Tuple[float, float],
        C: Tuple[float, float]) -> bool:
    """
    Test if three points are in counter-clockwise order.

    This function is a building block for line segment intersection testing.
    It determines the orientation of an ordered triplet of points.

    Args:
        A: First point (x, y)
        B: Second point (x, y)
        C: Third point (x, y)

    Returns:
        True if points A, B, C are in counter-clockwise order
        False otherwise (clockwise or collinear)

    Mathematical Background:
        Uses the cross product of vectors AB and AC:
        - Positive cross product → counter-clockwise
        - Negative cross product → clockwise
        - Zero cross product → collinear

    Example:
        >>> ccw((0, 0), (1, 1), (0, 2))
        True  # Counter-clockwise
        >>> ccw((0, 0), (1, 1), (2, 0))
        False  # Clockwise
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Tuple[float, float],
              B: Tuple[float, float],
              C: Tuple[float, float],
              D: Tuple[float, float]) -> bool:
    """
    Test if line segment AB intersects with line segment CD.

    This is a robust implementation using the counter-clockwise (CCW) test.
    Two segments intersect if and only if:
    1. The endpoints of one segment are on opposite sides of the other segment
    2. The endpoints of the other segment are on opposite sides of the first

    Args:
        A: Start of first line segment (x, y)
        B: End of first line segment (x, y)
        C: Start of second line segment (x, y)
        D: End of second line segment (x, y)

    Returns:
        True if segments AB and CD intersect
        False if segments do not intersect

    Algorithm:
        Uses CCW orientation test:
        - Segments intersect if: ccw(A,C,D) ≠ ccw(B,C,D) AND ccw(A,B,C) ≠ ccw(A,B,D)
        - This checks if C and D are on opposite sides of AB,
          AND if A and B are on opposite sides of CD

    Examples:
        >>> # Crossing segments
        >>> intersect((0, 0), (2, 2), (0, 2), (2, 0))
        True

        >>> # Non-crossing segments
        >>> intersect((0, 0), (1, 1), (2, 2), (3, 3))
        False

        >>> # Parallel segments
        >>> intersect((0, 0), (1, 0), (0, 1), (1, 1))
        False

    Note:
        This function does NOT detect:
        - Overlapping collinear segments as intersecting
        - Touching endpoints as intersecting
        For strict intersection testing, this is the desired behavior.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
