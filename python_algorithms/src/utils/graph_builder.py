"""
Graph construction utilities for graph-based path planning algorithms.

This module provides functions to build visibility graphs from obstacle corners
and compute adjacency matrices with collision-free edges.

Used by: A* and Dijkstra algorithms
"""

import math
from typing import List, Tuple
from .geometry import intersect


def create_edges(source: Tuple[float, float],
                 goal: Tuple[float, float],
                 centers: List[Tuple[float, float]],
                 sides: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Create graph nodes from start, goal, and obstacle corners.

    This function generates a visibility graph by including:
    - The source (start) point
    - All four corners of each rectangular obstacle
    - The goal point

    Args:
        source: Starting position (x, y)
        goal: Goal position (x, y)
        centers: List of obstacle center points [(x1, y1), (x2, y2), ...]
        sides: List of obstacle dimensions [(w1, h1), (w2, h2), ...]
               Each (w, h) corresponds to the obstacle at the same index in centers

    Returns:
        List of all graph nodes (coordinates):
        [source, corner1, corner2, ..., cornerN, goal]

    Example:
        >>> source = (0, 0)
        >>> goal = (10, 10)
        >>> centers = [(5, 5)]
        >>> sides = [(2, 2)]
        >>> nodes = create_edges(source, goal, centers, sides)
        >>> len(nodes)  # 1 source + 4 corners + 1 goal = 6
        6

    Note:
        The order is important for indexing in the adjacency matrix:
        - Node 0: source
        - Nodes 1 to 4*N: obstacle corners (4 per obstacle)
        - Node 4*N+1: goal
    """
    coordinates = [source]

    # Add all four corners of each obstacle
    for center, side in zip(centers, sides):
        cx, cy = center
        w, h = side

        # Four corners: bottom-left, bottom-right, top-right, top-left
        coordinates.extend([
            (cx - 0.5 * w, cy + 0.5 * h),  # Top-left
            (cx + 0.5 * w, cy + 0.5 * h),  # Top-right
            (cx - 0.5 * w, cy - 0.5 * h),  # Bottom-left
            (cx + 0.5 * w, cy - 0.5 * h),  # Bottom-right
        ])

    coordinates.append(goal)
    return coordinates


def create_adjacency_matrix(coordinates: List[Tuple[float, float]],
                            centers: List[Tuple[float, float]],
                            sides: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Build adjacency matrix for visibility graph with collision checking.

    For each pair of nodes, this function:
    1. Checks if the straight line between them intersects any obstacle
    2. If collision-free, stores the Euclidean distance as edge weight
    3. If collision or same node, stores infinity (no edge)

    Args:
        coordinates: List of all graph nodes from create_edges()
        centers: List of obstacle center points
        sides: List of obstacle dimensions

    Returns:
        2D adjacency matrix where graph[i][j] is:
        - 0.0 if i == j (distance to self)
        - Euclidean distance if edge (i,j) is collision-free
        - float('inf') if edge would collide with an obstacle

    Algorithm:
        For each pair of nodes (i, j):
        1. Construct line segment from coordinates[i] to coordinates[j]
        2. For each rectangular obstacle:
           - Get the 4 edges of the rectangle
           - Check if line segment intersects any of the 4 edges
           - If intersection found, mark edge as blocked (infinity)
        3. If no collisions, compute and store Euclidean distance

    Example:
        >>> coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        >>> centers = [(5, 5)]
        >>> sides = [(2, 2)]
        >>> graph = create_adjacency_matrix(coords, centers, sides)
        >>> graph[0][1]  # Distance from node 0 to node 1
        10.0
        >>> graph[0][2]  # Distance from node 0 to node 2 (may be blocked)
        inf  # If obstacle is in the way

    Performance:
        - Time complexity: O(N² * M) where N = nodes, M = obstacles
        - Space complexity: O(N²)
    """
    num_nodes = len(coordinates)

    # Initialize matrix with infinity (no edges)
    graph = [[float('inf')] * num_nodes for _ in range(num_nodes)]

    # Distance from node to itself is 0
    for i in range(num_nodes):
        graph[i][i] = 0

    # Check each pair of nodes for collision-free edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            collision = False

            # Check intersection with each obstacle
            for rect_center, rect_side in zip(centers, sides):
                cx, cy = rect_center
                w, h = rect_side

                # Four corners of the rectangle
                rect_x1 = cx - 0.5 * w
                rect_y1 = cy - 0.5 * h
                rect_x2 = cx + 0.5 * w
                rect_y2 = cy + 0.5 * h

                # Four edges of the rectangle
                # Edge 1: bottom edge (left to right)
                # Edge 2: right edge (bottom to top)
                # Edge 3: top edge (right to left)
                # Edge 4: left edge (top to bottom)
                if (intersect((x1, y1), (x2, y2), (rect_x1, rect_y1), (rect_x2, rect_y1)) or
                    intersect((x1, y1), (x2, y2), (rect_x2, rect_y1), (rect_x2, rect_y2)) or
                    intersect((x1, y1), (x2, y2), (rect_x2, rect_y2), (rect_x1, rect_y2)) or
                    intersect((x1, y1), (x2, y2), (rect_x1, rect_y2), (rect_x1, rect_y1))):
                    collision = True
                    break

            # If no collision, add edge with Euclidean distance
            if not collision:
                dist = math.dist(coordinates[i], coordinates[j])
                graph[i][j] = dist
                graph[j][i] = dist  # Undirected graph (symmetric)

    return graph
