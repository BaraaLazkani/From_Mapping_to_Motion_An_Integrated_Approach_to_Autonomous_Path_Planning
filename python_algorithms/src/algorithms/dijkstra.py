"""
Dijkstra's shortest path algorithm implementation.

Dijkstra's algorithm finds the shortest path in a weighted graph without using
a heuristic, exploring nodes in order of increasing distance from the start.
"""

import heapq
import time
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.animation import FuncAnimation

from ..core.path_planner import PathPlanner
from ..core.trajectory import CSCTrajectory
from ..utils.graph_builder import create_edges, create_adjacency_matrix
from ..utils.visualization import draw_environment, draw_robot


class DijkstraPlanner(PathPlanner):
    """
    Dijkstra's shortest path algorithm.

    Unlike A*, Dijkstra explores nodes without a heuristic, guaranteeing
    the shortest path by expanding nodes in order of cumulative cost from start.

    Attributes:
        graph (List[List[float]]): Adjacency matrix of the visibility graph
        coordinates (List[Tuple[float, float]]): Node coordinates
        previous_nodes (Dict): Parent mapping for path reconstruction
        nodes_explored (int): Number of nodes examined
    """

    def _initialize_algorithm(self) -> None:
        """Initialize Dijkstra-specific data structures."""
        self.graph = None
        self.coordinates = None
        self.previous_nodes = {}
        self.nodes_explored = 0

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Compute shortest path using Dijkstra's algorithm.

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

        num_nodes = len(self.graph)
        start_idx = 0
        goal_idx = num_nodes - 1

        # Initialize distances
        distances = {node: float('inf') for node in range(num_nodes)}
        distances[start_idx] = 0

        self.previous_nodes = {node: None for node in range(num_nodes)}

        # Priority queue: (distance, node)
        priority_queue = [(0, start_idx)]
        visited = set()

        # Dijkstra's main loop
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            self.nodes_explored += 1

            if current_node in visited:
                continue

            visited.add(current_node)

            # Goal reached
            if current_node == goal_idx:
                break

            # Explore neighbors
            for neighbor in range(num_nodes):
                if self.graph[current_node][neighbor] == float('inf') or neighbor in visited:
                    continue

                new_distance = current_distance + self.graph[current_node][neighbor]

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    self.previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        # Reconstruct path
        path_indices = []
        current = goal_idx

        while current is not None:
            path_indices.append(current)
            current = self.previous_nodes[current]

        if path_indices and path_indices[-1] == start_idx:
            path_indices = path_indices[::-1]  # Reverse to get start-to-goal
            self.path = [self.coordinates[i] for i in path_indices]
        else:
            self.path = None  # No path found

        self.planning_time = time.time() - start_time
        return self.path

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get algorithm performance metrics.

        Returns:
            Dictionary with algorithm statistics
        """
        return {
            'algorithm': 'Dijkstra',
            'path_length': self.get_path_length(),
            'planning_time': self.planning_time,
            'nodes_explored': self.nodes_explored,
            'path_exists': self.path is not None
        }

    def visualize(self, ax, **kwargs) -> None:
        """
        Visualize Dijkstra's path.

        Args:
            ax: Matplotlib axis
            **kwargs: Additional visualization options
        """
        if self.coordinates is None:
            return

        # Find path indices
        path_indices = []
        if self.path:
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
            path_color=kwargs.get('path_color', 'green'),
            path_label="Dijkstra Path"
        )

        ax.set_title(f"Dijkstra's Algorithm\n"
                    f"Length: {self.get_path_length():.2f}, "
                    f"Time: {self.planning_time:.3f}s, "
                    f"Nodes: {self.nodes_explored}")

    def animate_trajectory(self, fig, ax, save_path: Optional[str] = None) -> Optional[FuncAnimation]:
        """
        Create animation of robot following Dijkstra path with CSC smoothing.

        Args:
            fig: Matplotlib figure
            ax: Matplotlib axis
            save_path: Optional path to save animation

        Returns:
            FuncAnimation object if successful, None otherwise
        """
        if self.path is None or len(self.path) < 2:
            print("No path to animate")
            return None

        traj_config = self.config.get('trajectory', {})
        if not traj_config.get('enabled', True):
            print("Trajectory smoothing disabled")
            return None

        robot_traj = CSCTrajectory(
            R=traj_config.get('radius', 1.0),
            r_width=traj_config.get('robot_width', 0.5)
        )

        current_index = 0
        t = 0
        robot_x, robot_y = self.path[0]
        robot_theta = 0.0
        segments_per_edge = traj_config.get('segments_per_edge', 30)

        def animate(frame):
            nonlocal current_index, t, robot_x, robot_y, robot_theta

            if current_index < len(self.path) - 1:
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

            self.visualize(ax)
            draw_robot(robot_x, robot_y, robot_theta, ax)

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
