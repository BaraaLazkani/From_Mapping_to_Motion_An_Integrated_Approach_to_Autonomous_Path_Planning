"""
Node class for graph-based path planning algorithms.

Simple node structure used by RRT* algorithm.
"""


class Node:
    """
    Represents a node in the search tree/graph.

    Used primarily by RRT* algorithm for tree construction.

    Attributes:
        x (float): X-coordinate
        y (float): Y-coordinate
        parent (Node): Parent node in the tree (None for root)
        cost (float): Cost from root to this node
    """

    def __init__(self, x: float, y: float):
        """
        Initialize a node at given coordinates.

        Args:
            x: X-coordinate
            y: Y-coordinate
        """
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.2f})"
