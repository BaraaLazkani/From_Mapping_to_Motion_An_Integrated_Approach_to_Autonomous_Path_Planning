"""
RRT* (Rapidly-exploring Random Tree Star) algorithm implementation.

RRT* is a sampling-based path planning algorithm that builds a tree by
randomly sampling the space and includes rewiring for asymptotic optimality.
"""

import random
import math
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.animation import FuncAnimation

from ..core.path_planner import PathPlanner
from ..core.node import Node
from ..utils.geometry import intersect
from ..utils.visualization import draw_environment


class RRTStarPlanner(PathPlanner):
    """
    RRT* path planning algorithm.

    Builds a tree by random sampling, steering toward samples, and
    rewiring the tree to improve path quality over time.

    Attributes:
        start_node (Node): Root of the tree
        goal_node (Node): Goal position as a node
        node_list (List[Node]): All nodes in the tree
        goal_reached (bool): Whether goal has been connected to tree
        step_size (float): Maximum edge length when extending tree
        max_iter (int): Maximum number of samples
        rewire_radius (float): Radius for finding rewiring candidates
        goal_tolerance (float): Distance to consider goal reached
    """

    def _initialize_algorithm(self) -> None:
        """Initialize RRT*-specific data structures."""
        self.start_node = None
        self.goal_node = None
        self.node_list = []
        self.goal_reached = False
        self.nodes_explored = 0

        # Get algorithm parameters from config
        params = self.config.get('parameters', {})
        self.step_size = params.get('step_size', 0.5)
        self.max_iter = params.get('max_iterations', 500)
        self.rewire_radius = params.get('rewire_radius', 2.0)
        self.goal_tolerance = params.get('goal_tolerance', 1.0)
        self.goal_sample_rate = params.get('goal_sample_rate', 0.2)

        # Set random seed if specified
        seed = params.get('random_seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_random_node(self) -> Node:
        """
        Generate a random node for sampling.

        With probability goal_sample_rate, samples the goal.
        Otherwise samples uniformly in the environment.

        Returns:
            Randomly sampled node
        """
        if random.random() < self.goal_sample_rate:
            return Node(self.goal_node.x, self.goal_node.y)
        else:
            x = random.uniform(0, self.environment.map_size[0])
            y = random.uniform(0, self.environment.map_size[1])
            return Node(x, y)

    def _get_nearest_node(self, target_node: Node) -> Node:
        """
        Find nearest node in tree to target.

        Args:
            target_node: Node to find nearest neighbor for

        Returns:
            Nearest node in the tree
        """
        distances = [math.hypot(node.x - target_node.x, node.y - target_node.y)
                    for node in self.node_list]
        nearest_idx = distances.index(min(distances))
        return self.node_list[nearest_idx]

    def _steer(self, from_node: Node, to_node: Node) -> Node:
        """
        Steer from from_node toward to_node by at most step_size.

        Args:
            from_node: Starting node
            to_node: Target node

        Returns:
            New node at most step_size away from from_node in direction of to_node
        """
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(
            from_node.x + self.step_size * math.cos(theta),
            from_node.y + self.step_size * math.sin(theta)
        )
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.step_size
        return new_node

    def _is_collision_free(self, node: Node, parent_node: Node) -> bool:
        """
        Check if edge from parent to node is collision-free.

        Args:
            node: End node
            parent_node: Start node

        Returns:
            True if edge doesn't collide with obstacles
        """
        return self.environment.is_edge_collision_free(
            (parent_node.x, parent_node.y),
            (node.x, node.y)
        )

    def _reached_goal(self, node: Node) -> bool:
        """
        Check if node is close enough to goal.

        Args:
            node: Node to check

        Returns:
            True if within goal_tolerance of goal
        """
        dist = math.hypot(node.x - self.goal_node.x, node.y - self.goal_node.y)
        return dist < self.goal_tolerance

    def _find_near_nodes(self, new_node: Node) -> List[Node]:
        """
        Find all nodes within rewire_radius of new_node.

        Args:
            new_node: Node to find neighbors for

        Returns:
            List of nearby nodes
        """
        near_nodes = []
        for node in self.node_list:
            dist = math.hypot(node.x - new_node.x, node.y - new_node.y)
            if dist <= self.rewire_radius:
                near_nodes.append(node)
        return near_nodes

    def _choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        """
        Choose best parent for new_node from near_nodes.

        Selects parent that minimizes cost to reach new_node.

        Args:
            new_node: Node to find parent for
            near_nodes: Candidate parent nodes

        Returns:
            Best parent node
        """
        if not near_nodes:
            return new_node.parent

        costs = []
        for near_node in near_nodes:
            if self._is_collision_free(new_node, near_node):
                cost = near_node.cost + math.hypot(new_node.x - near_node.x,
                                                   new_node.y - near_node.y)
                costs.append(cost)
            else:
                costs.append(float('inf'))

        min_cost_idx = costs.index(min(costs))
        if costs[min_cost_idx] != float('inf'):
            new_node.parent = near_nodes[min_cost_idx]
            new_node.cost = costs[min_cost_idx]

        return new_node.parent

    def _rewire(self, new_node: Node, near_nodes: List[Node]) -> None:
        """
        Rewire tree by checking if paths through new_node are better.

        Args:
            new_node: Newly added node
            near_nodes: Nodes to potentially rewire
        """
        for near_node in near_nodes:
            if near_node == new_node or near_node.parent is None:
                continue

            edge_cost = math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            new_cost = new_node.cost + edge_cost

            if new_cost < near_node.cost and self._is_collision_free(near_node, new_node):
                near_node.parent = new_node
                near_node.cost = new_cost

    def _generate_final_path(self, last_node: Node) -> List[Tuple[float, float]]:
        """
        Extract path from start to last_node by following parent pointers.

        Args:
            last_node: Goal node or closest node to goal

        Returns:
            List of waypoints from start to last_node
        """
        path = []
        node = last_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]  # Reverse to get start-to-goal order

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Build RRT* tree and find path to goal.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)

        Returns:
            List of waypoints if path found, None otherwise
        """
        start_time = time.time()

        # Initialize tree
        self.start_node = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.node_list = [self.start_node]
        self.goal_reached = False
        self.nodes_explored = 0

        best_goal_node = None

        # Main RRT* loop
        for i in range(self.max_iter):
            # Sample random node
            rand_node = self._get_random_node()
            self.nodes_explored += 1

            # Find nearest node in tree
            nearest_node = self._get_nearest_node(rand_node)

            # Steer toward sample
            new_node = self._steer(nearest_node, rand_node)

            # Check collision
            if self._is_collision_free(new_node, nearest_node):
                # Find nearby nodes for rewiring
                near_nodes = self._find_near_nodes(new_node)

                # Choose best parent
                self._choose_parent(new_node, near_nodes)

                # Add to tree
                self.node_list.append(new_node)

                # Rewire tree
                self._rewire(new_node, near_nodes)

                # Check if goal reached
                if self._reached_goal(new_node):
                    self.goal_reached = True
                    if best_goal_node is None or new_node.cost < best_goal_node.cost:
                        best_goal_node = new_node

        # Generate path
        if best_goal_node is not None:
            self.path = self._generate_final_path(best_goal_node)
        else:
            self.path = None

        self.planning_time = time.time() - start_time
        return self.path

    def get_metrics(self) -> Dict[str, Any]:
        """Get RRT* performance metrics."""
        return {
            'algorithm': 'RRT*',
            'path_length': self.get_path_length(),
            'planning_time': self.planning_time,
            'nodes_explored': self.nodes_explored,
            'tree_size': len(self.node_list),
            'goal_reached': self.goal_reached,
            'path_exists': self.path is not None
        }

    def visualize(self, ax, show_tree: bool = True, **kwargs) -> None:
        """
        Visualize RRT* tree and path.

        Args:
            ax: Matplotlib axis
            show_tree: Whether to draw the full tree
            **kwargs: Additional options
        """
        # Draw tree edges
        if show_tree:
            tree_color = self.config.get('visualization', {}).get('tree_color', 'blue')
            tree_alpha = self.config.get('visualization', {}).get('tree_alpha', 0.3)

            for node in self.node_list:
                if node.parent is not None:
                    ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                           color=tree_color, linewidth=0.5, alpha=tree_alpha, zorder=1)

        # Draw path
        if self.path:
            path_x, path_y = zip(*self.path)
            path_color = self.config.get('visualization', {}).get('path_color', 'green')
            ax.plot(path_x, path_y, color=path_color, linewidth=2, label="RRT* Path", zorder=3)

        # Draw environment
        draw_environment(
            ax,
            self.environment.obstacle_centers,
            self.environment.obstacle_sizes,
            start=self.path[0] if self.path else None,
            goal=self.path[-1] if self.path else None
        )

        ax.set_title(f"RRT* Algorithm\n"
                    f"Length: {self.get_path_length():.2f}, "
                    f"Time: {self.planning_time:.3f}s, "
                    f"Nodes: {len(self.node_list)}")
