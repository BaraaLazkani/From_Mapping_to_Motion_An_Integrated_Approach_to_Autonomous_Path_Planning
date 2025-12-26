"""
A* pathfinding algorithm implementation.

A* is an informed search algorithm that uses a heuristic to efficiently find
optimal paths in a graph.
"""

import heapq
import math
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.animation import FuncAnimation

from ..core.path_planner import PathPlanner
from ..core.trajectory import CSCTrajectory
from ..utils.graph_builder import create_edges, create_adjacency_matrix
from ..utils.visualization import draw_environment, draw_robot


class AStarPlanner(PathPlanner):
    """
    A* path planning algorithm.

    A* uses a heuristic function to guide the search toward the goal,
    guaranteeing optimal paths when the heuristic is admissible.

    The algorithm builds a visibility graph from obstacle corners and
    searches for the shortest collision-free path.

    Attributes:
        graph (List[List[float]]): Adjacency matrix of the visibility graph
        coordinates (List[Tuple[float, float]]): Node coordinates in the graph
        came_from (Dict): Parent mapping for path reconstruction
        nodes_explored (int): Number of nodes expanded during search
    """

    def _initialize_algorithm(self) -> None:
        """Initialize A*-specific data structures."""
        self.graph = None
        self.coordinates = None
        self.came_from = {}
        self.nodes_explored = 0

    def _heuristic(self, node_idx: int, goal_idx: int) -> float:
        """
        Compute heuristic estimate from node to goal.

        Uses Euclidean distance as the heuristic (admissible and consistent).

        Args:
            node_idx: Index of current node
            goal_idx: Index of goal node

        Returns:
            Estimated cost from node to goal
        """
        heuristic_type = self.config.get('parameters', {}).get('heuristic_type', 'euclidean')

        if heuristic_type == 'euclidean':
            return math.dist(self.coordinates[node_idx], self.coordinates[goal_idx])
        elif heuristic_type == 'manhattan':
            x1, y1 = self.coordinates[node_idx]
            x2, y2 = self.coordinates[goal_idx]
            return abs(x2 - x1) + abs(y2 - y1)
        else:  # Default to Euclidean
            return math.dist(self.coordinates[node_idx], self.coordinates[goal_idx])

    def _reconstruct_path(self, current_idx: int) -> List[int]:
        """
        Reconstruct path from goal to start using parent pointers.

        Args:
            current_idx: Index of goal node

        Returns:
            List of node indices from start to goal
        """
        path = [current_idx]
        while current_idx in self.came_from:
            current_idx = self.came_from[current_idx]
            path.append(current_idx)
        return path[::-1]  # Reverse to get start-to-goal order

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Compute shortest path from start to goal using A*.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)

        Returns:
            List of waypoints if path found, None otherwise
        """
        start_time = time.time()
        self.nodes_explored = 0

        # Build visibility graph
        obstacle_centers = [(c[0], c[1]) for c in self.environment.obstacle_centers]
        obstacle_sizes = [(s[0], s[1]) for s in self.environment.obstacle_sizes]

        self.coordinates = create_edges(start, goal, obstacle_centers, obstacle_sizes)
        self.graph = create_adjacency_matrix(self.coordinates, obstacle_centers, obstacle_sizes)

        # A* search
        start_idx = 0
        goal_idx = len(self.coordinates) - 1

        # Priority queue: (f_score, node_index)
        open_set = [(0, start_idx)]
        heapq.heapify(open_set)

        # Track costs
        g_score = {node: float('inf') for node in range(len(self.graph))}
        g_score[start_idx] = 0

        f_score = {node: float('inf') for node in range(len(self.graph))}
        f_score[start_idx] = self._heuristic(start_idx, goal_idx)

        self.came_from = {}

        # Main A* loop
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            self.nodes_explored += 1

            # Goal reached
            if current_node == goal_idx:
                path_indices = self._reconstruct_path(current_node)
                self.path = [self.coordinates[i] for i in path_indices]
                self.planning_time = time.time() - start_time
                return self.path

            # Explore neighbors
            for neighbor in range(len(self.graph)):
                if self.graph[current_node][neighbor] == float('inf'):
                    continue  # No edge

                tentative_g_score = g_score[current_node] + self.graph[current_node][neighbor]

                if tentative_g_score < g_score[neighbor]:
                    # Better path found
                    self.came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        self.planning_time = time.time() - start_time
        self.path = None
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get algorithm performance metrics.

        Returns:
            Dictionary with:
            - path_length: Total path length
            - planning_time: Time to compute path
            - nodes_explored: Number of nodes examined
        """
        return {
            'algorithm': 'A*',
            'path_length': self.get_path_length(),
            'planning_time': self.planning_time,
            'nodes_explored': self.nodes_explored,
            'path_exists': self.path is not None
        }

    def visualize(self, ax, **kwargs) -> None:
        """
        Visualize the A* path on matplotlib axis.

        Args:
            ax: Matplotlib axis
            **kwargs: Additional visualization parameters
        """
        if self.coordinates is None:
            return

        # Draw environment with path
        path_indices = []
        if self.path:
            # Find indices of path coordinates
            for waypoint in self.path:
                for idx, coord in enumerate(self.coordinates):
                    if abs(coord[0] - waypoint[0]) < 0.01 and abs(coord[1] - waypoint[1]) < 0.01:
                        path_indices.append(idx)
                        break

        draw_environment(
            ax,
            self.environment.obstacle_centers,
            self.environment.obstacle_sizes,
            path=path_indices if path_indices else None,
            coordinates=self.coordinates,
            start=self.coordinates[0] if self.coordinates else None,
            goal=self.coordinates[-1] if self.coordinates else None,
            path_color=kwargs.get('path_color', 'blue'),
            path_label="A* Path"
        )

        ax.set_title(f"A* Path Planning\n"
                    f"Length: {self.get_path_length():.2f}, "
                    f"Time: {self.planning_time:.3f}s, "
                    f"Nodes: {self.nodes_explored}")

    def animate_trajectory(self, fig, ax, save_path: Optional[str] = None) -> Optional[FuncAnimation]:
        """
        Create animation of robot following the A* path with CSC trajectory smoothing.

        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            save_path: Optional path to save animation GIF

        Returns:
            FuncAnimation object if path exists, None otherwise
        """
        if self.path is None or len(self.path) < 2:
            print("No path to animate")
            return None

        # Initialize CSC trajectory smoother
        traj_config = self.config.get('trajectory', {})
        if not traj_config.get('enabled', True):
            print("Trajectory smoothing disabled")
            return None

        robot_traj = CSCTrajectory(
            R=traj_config.get('radius', 1.0),
            r_width=traj_config.get('robot_width', 0.5)
        )

        # Animation state
        current_index = 0
        t = 0
        robot_x, robot_y = self.path[0]
        robot_theta = 0.0
        segments_per_edge = traj_config.get('segments_per_edge', 30)

        def animate(frame):
            nonlocal current_index, t, robot_x, robot_y, robot_theta

            if current_index < len(self.path) - 1:
                # New segment
                if t == 0:
                    initial = self.path[current_index]
                    final = self.path[current_index + 1]
                    robot_traj.set_cond(
                        (initial[0], initial[1], robot_theta),
                        (final[0], final[1], 0)
                    )

                robot_x, robot_y, robot_theta = robot_traj.update_pos(t)
                t += 1

                if t > segments_per_edge:
                    t = 0
                    current_index += 1

            # Draw
            self.visualize(ax)
            draw_robot(robot_x, robot_y, robot_theta, ax)

            # Dynamic view
            margin = max(self.environment.map_size) * 0.3
            ax.set_xlim(robot_x - margin, robot_x + margin)
            ax.set_ylim(robot_y - margin, robot_y + margin)

        total_frames = len(self.path) * segments_per_edge
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)

        if save_path:
            anim.save(save_path, writer='pillow',
                     fps=self.config.get('output', {}).get('animation_fps', 5))
            print(f"Animation saved to {save_path}")

        return anim
