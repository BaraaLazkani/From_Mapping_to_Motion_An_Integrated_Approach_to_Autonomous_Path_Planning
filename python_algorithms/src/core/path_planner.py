"""
Abstract base class for all path planning algorithms.

This module defines the common interface that all path planning algorithms
(A*, Dijkstra, RRT*, APF) must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time


class PathPlanner(ABC):
    """
    Abstract base class for path planning algorithms.

    All path planning algorithms inherit from this class and implement
    the required abstract methods for planning, visualization, and metrics.

    Attributes:
        environment: The planning environment containing obstacles and bounds
        config (Dict[str, Any]): Algorithm-specific configuration parameters
        path (Optional[List[Tuple[float, float]]]): Computed path from start to goal
        planning_time (float): Time taken to compute the path (seconds)
        metrics (Dict[str, Any]): Performance metrics from last planning run
    """

    def __init__(self, environment, config: Dict[str, Any]):
        """
        Initialize the path planner.

        Args:
            environment: Environment object containing obstacles, bounds, etc.
            config: Dictionary of algorithm-specific parameters loaded from YAML
        """
        self.environment = environment
        self.config = config
        self.path: Optional[List[Tuple[float, float]]] = None
        self.planning_time: float = 0.0
        self.metrics: Dict[str, Any] = {}
        self._initialize_algorithm()

    @abstractmethod
    def _initialize_algorithm(self) -> None:
        """
        Initialize algorithm-specific data structures.

        This method is called during __init__ and should set up any
        algorithm-specific variables, data structures, or parameters.

        Subclasses must override this method.
        """
        pass

    @abstractmethod
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Compute a collision-free path from start to goal.

        This is the core planning method that must be implemented by each algorithm.
        The method should:
        1. Find a path from start to goal
        2. Ensure the path is collision-free
        3. Store the result in self.path
        4. Measure and store planning time in self.planning_time
        5. Return the computed path

        Args:
            start: Starting position (x, y) in environment coordinates
            goal: Goal position (x, y) in environment coordinates

        Returns:
            List of waypoints [(x1, y1), (x2, y2), ...] if path found
            None if no path exists

        Example:
            >>> planner = AStarPlanner(environment, config)
            >>> path = planner.plan((0, 0), (10, 10))
            >>> if path:
            >>>     print(f"Path found with {len(path)} waypoints")
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get algorithm performance metrics from the last planning run.

        Returns:
            Dictionary containing metrics such as:
            - path_length (float): Total Euclidean length of the path
            - planning_time (float): Time taken to compute path (seconds)
            - nodes_explored (int): Number of nodes/samples explored
            - algorithm_specific_metrics: Any additional algorithm-specific data

        Example:
            >>> metrics = planner.get_metrics()
            >>> print(f"Path length: {metrics['path_length']:.2f}")
            >>> print(f"Planning time: {metrics['planning_time']:.3f}s")
        """
        pass

    @abstractmethod
    def visualize(self, ax, **kwargs) -> None:
        """
        Visualize the planning result on a matplotlib axis.

        This method should draw the computed path, explored nodes,
        and any other algorithm-specific visualizations.

        Args:
            ax: Matplotlib axis object to draw on
            **kwargs: Additional visualization parameters (colors, line widths, etc.)

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> planner.visualize(ax, path_color='blue', linewidth=2)
            >>> plt.show()
        """
        pass

    def validate_path(self) -> bool:
        """
        Validate that the computed path is collision-free.

        Checks each edge in the path to ensure it doesn't collide with obstacles.

        Returns:
            True if path exists and is collision-free
            False if path doesn't exist or contains collisions
        """
        if self.path is None or len(self.path) < 2:
            return False

        # Check each edge in the path for collisions
        for i in range(len(self.path) - 1):
            if not self.environment.is_edge_collision_free(self.path[i], self.path[i+1]):
                return False

        return True

    def get_path_length(self) -> float:
        """
        Calculate the total Euclidean length of the computed path.

        Returns:
            Path length in environment units
            0.0 if no path exists

        Example:
            >>> path_length = planner.get_path_length()
            >>> print(f"Total distance: {path_length:.2f} meters")
        """
        if self.path is None or len(self.path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i+1][0] - self.path[i][0]
            dy = self.path[i+1][1] - self.path[i][1]
            total_length += np.sqrt(dx**2 + dy**2)

        return total_length

    def save_path(self, filename: str) -> None:
        """
        Save the computed path to a file.

        Supports multiple formats based on file extension:
        - .npy: NumPy binary format
        - .json: JSON format with path and metrics
        - .csv: Comma-separated values

        Args:
            filename: Output file path with extension

        Raises:
            ValueError: If no path exists or file format is unsupported

        Example:
            >>> planner.save_path('outputs/path.json')
            >>> planner.save_path('outputs/path.csv')
        """
        if self.path is None:
            raise ValueError("No path to save. Run plan() first.")

        if filename.endswith('.npy'):
            np.save(filename, np.array(self.path))
        elif filename.endswith('.json'):
            import json
            with open(filename, 'w') as f:
                json.dump({
                    'path': self.path,
                    'metrics': self.get_metrics()
                }, f, indent=2)
        elif filename.endswith('.csv'):
            np.savetxt(filename, np.array(self.path),
                      delimiter=',', header='x,y', comments='')
        else:
            raise ValueError(f"Unsupported file format: {filename}. "
                           f"Use .npy, .json, or .csv")

    def load_path(self, filename: str) -> List[Tuple[float, float]]:
        """
        Load a path from a file.

        Args:
            filename: Input file path

        Returns:
            List of waypoints loaded from file

        Example:
            >>> path = planner.load_path('outputs/path.npy')
        """
        if filename.endswith('.npy'):
            path_array = np.load(filename)
            self.path = [(x, y) for x, y in path_array]
        elif filename.endswith('.json'):
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
                self.path = [tuple(p) for p in data['path']]
        elif filename.endswith('.csv'):
            path_array = np.loadtxt(filename, delimiter=',', skiprows=1)
            self.path = [(x, y) for x, y in path_array]
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        return self.path

    def __repr__(self) -> str:
        """String representation of the planner."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "with path" if self.path else "no path"
        return f"{self.__class__.__name__} ({status})"
