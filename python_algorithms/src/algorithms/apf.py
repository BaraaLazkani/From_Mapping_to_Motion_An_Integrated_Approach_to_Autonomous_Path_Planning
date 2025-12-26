"""
Artificial Potential Field (APF) path planning algorithm.

APF uses attractive forces toward the goal and repulsive forces from obstacles
to navigate the robot to the goal.
"""

import time
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.animation import FuncAnimation

from ..core.path_planner import PathPlanner
from ..utils.visualization import draw_environment


class APFPlanner(PathPlanner):
    """
    Artificial Potential Field planner.

    Navigates by computing forces at each step:
    - Attractive force pulling toward goal
    - Repulsive forces pushing away from obstacles

    The robot follows the resultant force direction.

    Attributes:
        trajectory (List[Tuple[float, float]]): Recorded trajectory points
        obstacle_points (List[np.ndarray]): Discretized obstacle boundary points
        step_count (int): Number of simulation steps taken
    """

    def _initialize_algorithm(self) -> None:
        """Initialize APF-specific structures."""
        self.trajectory = []
        self.obstacle_points = []
        self.step_count = 0

        # Get parameters
        params = self.config.get('parameters', {})
        self.zeta = params.get('zeta', 1.1547)  # Attractive gain
        self.eta = params.get('eta', 0.0732)     # Repulsive gain
        self.dstar = params.get('dstar', 0.3)    # Attractive distance threshold
        self.Qstar = params.get('Qstar', 0.75)   # Repulsive influence radius
        self.max_v = params.get('max_linear_velocity', 0.2)
        self.max_omega = params.get('max_angular_velocity', np.pi/2)
        self.Kp_omega = params.get('Kp_omega', 1.5)
        self.error_theta_max = params.get('error_theta_max', np.deg2rad(45))
        self.dt = params.get('dt', 0.1)
        self.max_time = params.get('max_time', 1000)
        self.position_accuracy = params.get('position_accuracy', 0.05)

    def _discretize_obstacles(self) -> None:
        """
        Create discrete points along obstacle boundaries for force calculation.

        Generates evenly spaced points along each edge of rectangular obstacles.
        """
        points_per_edge = self.config.get('parameters', {}).get('obstacle_points_per_edge', 100)
        self.obstacle_points = []

        for center, size in zip(self.environment.obstacle_centers, self.environment.obstacle_sizes):
            cx, cy = center
            w, h = size

            # Obstacle corners
            x_min = cx - w / 2
            x_max = cx + w / 2
            y_min = cy - h / 2
            y_max = cy + h / 2

            # Create points along four edges
            points = np.vstack((
                np.concatenate([
                    np.linspace(x_min, x_max, points_per_edge),  # Top edge
                    np.full(points_per_edge, x_max),              # Right edge
                    np.linspace(x_max, x_min, points_per_edge),  # Bottom edge
                    np.full(points_per_edge, x_min)               # Left edge
                ]),
                np.concatenate([
                    np.full(points_per_edge, y_max),              # Top edge
                    np.linspace(y_max, y_min, points_per_edge),  # Right edge
                    np.full(points_per_edge, y_min),              # Bottom edge
                    np.linspace(y_min, y_max, points_per_edge)   # Left edge
                ])
            ))

            self.obstacle_points.append(points)

    def _closest_obstacle_point(self, x: float, y: float) -> Tuple[float, np.ndarray]:
        """
        Find closest point on any obstacle to (x, y).

        Args:
            x: Query x-coordinate
            y: Query y-coordinate

        Returns:
            Tuple of (min_distance, closest_point_coordinates)
        """
        min_dist = float('inf')
        closest_point = None

        for obst_points in self.obstacle_points:
            distances = np.sqrt((obst_points[0, :] - x)**2 + (obst_points[1, :] - y)**2)
            min_idx = np.argmin(distances)
            dist = distances[min_idx]

            if dist < min_dist:
                min_dist = dist
                closest_point = obst_points[:, min_idx]

        return min_dist, closest_point

    def _compute_attractive_force(self, x: float, y: float, goal_x: float, goal_y: float) -> np.ndarray:
        """
        Compute attractive potential gradient (force toward goal).

        Uses parabolic well within dstar, conic outside.

        Args:
            x: Current x-position
            y: Current y-position
            goal_x: Goal x-position
            goal_y: Goal y-position

        Returns:
            Attractive force vector [Fx, Fy]
        """
        dist_to_goal = np.linalg.norm([x - goal_x, y - goal_y])

        if dist_to_goal <= self.dstar:
            # Parabolic well (linear force)
            nabla_U_att = self.zeta * np.array([x - goal_x, y - goal_y])
        else:
            # Conic well (constant magnitude)
            nabla_U_att = (self.dstar / dist_to_goal) * self.zeta * np.array([x - goal_x, y - goal_y])

        return nabla_U_att

    def _compute_repulsive_force(self, x: float, y: float) -> np.ndarray:
        """
        Compute repulsive potential gradient (force away from obstacles).

        Sums repulsive forces from all nearby obstacles.

        Args:
            x: Current x-position
            y: Current y-position

        Returns:
            Repulsive force vector [Fx, Fy]
        """
        nabla_U_rep = np.array([0.0, 0.0])

        for obst_points in self.obstacle_points:
            obst_dist, obst_closest = self._closest_obstacle_point(x, y)

            if obst_dist <= self.Qstar:
                # Repulsive force inversely proportional to squared distance
                force_magnitude = self.eta * (1 / self.Qstar - 1 / obst_dist) * (1 / obst_dist ** 2)
                force_direction = np.array([x, y]) - obst_closest
                nabla_U_rep += force_magnitude * force_direction

        return nabla_U_rep

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Navigate from start to goal using artificial potential fields.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)

        Returns:
            Trajectory as list of waypoints
        """
        start_time = time.time()

        # Discretize obstacles
        self._discretize_obstacles()

        # Initialize state
        x, y = start
        theta = 0.0
        goal_x, goal_y = goal

        self.trajectory = [(x, y)]
        self.step_count = 0

        # Main APF loop
        while (np.linalg.norm([goal_x - x, goal_y - y]) > self.position_accuracy and
               self.step_count < self.max_time):

            # Compute potential forces
            F_att = self._compute_attractive_force(x, y, goal_x, goal_y)
            F_rep = self._compute_repulsive_force(x, y)
            F_total = F_att + F_rep

            # Compute reference heading
            theta_ref = np.arctan2(-F_total[1], -F_total[0])
            error_theta = theta_ref - theta
            # Normalize angle to [-pi, pi]
            error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))

            # Compute linear velocity (reduce if large heading error)
            if abs(error_theta) <= self.error_theta_max:
                alpha = (self.error_theta_max - abs(error_theta)) / self.error_theta_max
                v_ref = min(alpha * np.linalg.norm(-F_total), self.max_v)
            else:
                v_ref = 0.0  # Stop if heading error too large

            # Compute angular velocity (proportional control)
            omega_ref = self.Kp_omega * error_theta
            omega_ref = np.clip(omega_ref, -self.max_omega, self.max_omega)

            # Update state
            theta += omega_ref * self.dt * 8  # Scale factor for faster rotation
            x += v_ref * np.cos(theta) * self.dt * 10  # Scale factor for faster motion
            y += v_ref * np.sin(theta) * self.dt * 10

            # Record trajectory
            self.trajectory.append((x, y))
            self.step_count += 1

        self.path = self.trajectory
        self.planning_time = time.time() - start_time

        # Check if goal reached
        if np.linalg.norm([goal_x - x, goal_y - y]) <= self.position_accuracy:
            return self.path
        else:
            print(f"APF did not reach goal. Final distance: {np.linalg.norm([goal_x - x, goal_y - y]):.3f}")
            return self.path  # Return partial path

    def get_metrics(self) -> Dict[str, Any]:
        """Get APF performance metrics."""
        goal_dist = 0.0
        if self.path and len(self.path) > 0:
            final_pos = self.path[-1]
            # Would need goal position stored - for now return 0
            goal_dist = 0.0  # TODO: Store goal position

        return {
            'algorithm': 'APF',
            'path_length': self.get_path_length(),
            'planning_time': self.planning_time,
            'steps_taken': self.step_count,
            'trajectory_points': len(self.trajectory),
            'path_exists': self.path is not None
        }

    def visualize(self, ax, **kwargs) -> None:
        """
        Visualize APF trajectory.

        Args:
            ax: Matplotlib axis
            **kwargs: Additional visualization options
        """
        if not self.trajectory:
            return

        draw_environment(
            ax,
            self.environment.obstacle_centers,
            self.environment.obstacle_sizes,
            start=self.trajectory[0],
            goal=self.trajectory[-1]
        )

        # Draw trajectory
        traj_x, traj_y = zip(*self.trajectory)
        ax.plot(traj_x, traj_y, 'b-', linewidth=1.5, label="APF Trajectory", alpha=0.7)

        ax.set_title(f"Artificial Potential Field\n"
                    f"Length: {self.get_path_length():.2f}, "
                    f"Time: {self.planning_time:.3f}s, "
                    f"Steps: {self.step_count}")
