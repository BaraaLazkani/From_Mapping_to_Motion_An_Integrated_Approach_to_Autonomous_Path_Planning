"""
Environment representation for path planning.

This module defines the Environment class which encapsulates the planning space,
including obstacles, boundaries, and collision checking functionality.
"""

from typing import List, Tuple
import numpy as np


class Environment:
    """
    Represents the planning environment with obstacles and boundaries.

    The environment contains rectangular obstacles and provides methods for
    collision detection.

    Attributes:
        map_size (Tuple[float, float]): (width, height) of the environment
        obstacle_centers (List[Tuple[float, float]]): Centers of rectangular obstacles
        obstacle_sizes (List[Tuple[float, float]]): (width, height) of each obstacle
        bounds (Tuple[float, float, float, float]): (x_min, y_min, x_max, y_max)
    """

    def __init__(self,
                 map_size: Tuple[float, float],
                 obstacle_centers: List[Tuple[float, float]],
                 obstacle_sizes: List[Tuple[float, float]]):
        """
        Initialize the environment.

        Args:
            map_size: (width, height) of the environment
            obstacle_centers: List of obstacle centers [(x1, y1), (x2, y2), ...]
            obstacle_sizes: List of obstacle dimensions [(w1, h1), (w2, h2), ...]
                          corresponding to each obstacle in obstacle_centers

        Example:
            >>> env = Environment(
            ...     map_size=(20.0, 20.0),
            ...     obstacle_centers=[(5.0, 5.0), (12.0, 12.0)],
            ...     obstacle_sizes=[(2.0, 2.0), (3.0, 3.0)]
            ... )
        """
        self.map_size = map_size
        self.obstacle_centers = obstacle_centers
        self.obstacle_sizes = obstacle_sizes

        # Calculate environment bounds
        self.bounds = (0.0, 0.0, map_size[0], map_size[1])

    def is_point_in_bounds(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is within environment boundaries.

        Args:
            point: (x, y) coordinates

        Returns:
            True if point is within bounds, False otherwise
        """
        x, y = point
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def is_point_in_obstacle(self, point: Tuple[float, float], margin: float = 0.0) -> bool:
        """
        Check if a point is inside any obstacle.

        Args:
            point: (x, y) coordinates to check
            margin: Safety margin to add around obstacles (default: 0.0)

        Returns:
            True if point is inside an obstacle (including margin), False otherwise
        """
        x, y = point

        for center, size in zip(self.obstacle_centers, self.obstacle_sizes):
            cx, cy = center
            w, h = size

            # Add margin to obstacle boundaries
            x_min = cx - (w / 2.0) - margin
            x_max = cx + (w / 2.0) + margin
            y_min = cy - (h / 2.0) - margin
            y_max = cy + (h / 2.0) + margin

            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True

        return False

    def is_edge_collision_free(self,
                                 point1: Tuple[float, float],
                                 point2: Tuple[float, float]) -> bool:
        """
        Check if the straight line between two points is collision-free.

        Uses line segment intersection testing with obstacle edges.

        Args:
            point1: Start point (x1, y1)
            point2: End point (x2, y2)

        Returns:
            True if edge doesn't intersect any obstacles, False otherwise

        Note:
            This method uses the geometry utilities (ccw, intersect) for
            accurate intersection detection.
        """
        from ..utils.geometry import intersect

        # Check if either endpoint is in an obstacle
        if self.is_point_in_obstacle(point1) or self.is_point_in_obstacle(point2):
            return False

        # Check intersection with each obstacle's edges
        for center, size in zip(self.obstacle_centers, self.obstacle_sizes):
            cx, cy = center
            w, h = size

            # Four corners of the rectangle
            corners = [
                (cx - w/2, cy - h/2),  # bottom-left
                (cx + w/2, cy - h/2),  # bottom-right
                (cx + w/2, cy + h/2),  # top-right
                (cx - w/2, cy + h/2),  # top-left
            ]

            # Check intersection with each of the 4 edges
            for i in range(4):
                corner1 = corners[i]
                corner2 = corners[(i + 1) % 4]

                if intersect(point1, point2, corner1, corner2):
                    return False

        return True

    def get_obstacle_corners(self) -> List[Tuple[float, float]]:
        """
        Get all corner points of all obstacles.

        Useful for visibility graph construction in graph-based planners.

        Returns:
            List of corner points [(x1, y1), (x2, y2), ...]
        """
        corners = []

        for center, size in zip(self.obstacle_centers, self.obstacle_sizes):
            cx, cy = center
            w, h = size

            # Add all four corners
            corners.extend([
                (cx - w/2, cy - h/2),  # bottom-left
                (cx + w/2, cy - h/2),  # bottom-right
                (cx + w/2, cy + h/2),  # top-right
                (cx - w/2, cy + h/2),  # top-left
            ])

        return corners

    def __repr__(self) -> str:
        """String representation of the environment."""
        return (f"Environment(map_size={self.map_size}, "
                f"obstacles={len(self.obstacle_centers)})")
