"""
Dynamic Window Approach (DWA) Local Planner

The DWA algorithm samples velocity pairs (vL, vR) within the dynamic window
(velocities reachable in next time step) and scores each trajectory based on:
1. Forward progress toward goal
2. Obstacle clearance

The trajectory with highest score is selected for execution.
"""

import copy
import math
from typing import List, Tuple
from robot import DifferentialDriveRobot
from environment import Environment


class DWAPlanner:
    """Dynamic Window Approach local path planner."""

    def __init__(self, robot: DifferentialDriveRobot, environment: Environment,
                 timestep: float, steps_ahead: int, forward_weight: float,
                 obstacle_weight: float):
        """
        Initialize DWA planner.

        Args:
            robot: Robot instance
            environment: Environment instance
            timestep: Planning timestep (seconds)
            steps_ahead: Number of steps to simulate ahead
            forward_weight: Weight for forward progress toward goal
            obstacle_weight: Penalty weight for proximity to obstacles
        """
        self.robot = robot
        self.env = environment
        self.dt = timestep
        self.steps_ahead = steps_ahead
        self.tau = timestep * steps_ahead  # Total prediction time
        self.forward_weight = forward_weight
        self.obstacle_weight = obstacle_weight

        # For visualization
        self.paths_to_draw = []
        self.new_positions_to_draw = []

    def plan(self) -> Tuple[float, float]:
        """
        Execute one DWA planning cycle.

        Algorithm:
        1. Save current obstacle state
        2. Predict obstacle motion for planning horizon
        3. Sample velocity pairs within dynamic window
        4. For each velocity pair:
           - Simulate trajectory to end of planning horizon
           - Score based on goal approach and obstacle clearance
        5. Select velocity pair with best score
        6. Restore obstacle state

        Returns:
            Tuple of (vL_chosen, vR_chosen) for robot to execute
        """
        # Get current state
        x, y, theta = self.robot.get_pose()
        vL, vR = self.robot.get_velocities()
        target_x, target_y = self.env.get_target_position()

        # Save current obstacle state
        barriers_copy = copy.deepcopy(self.env.barriers)

        # Predict obstacle motion for planning horizon
        for _ in range(self.steps_ahead):
            self.env.update_obstacles(self.dt)

        # Sample velocities within dynamic window
        # Dynamic window: velocities reachable in next timestep given max acceleration
        vL_possible_array = (
            vL - self.robot.max_acceleration * self.dt,
            vL,
            vL + self.robot.max_acceleration * self.dt
        )
        vR_possible_array = (
            vR - self.robot.max_acceleration * self.dt,
            vR,
            vR + self.robot.max_acceleration * self.dt
        )

        # Evaluate all velocity combinations
        best_benefit = -100000.0
        vL_chosen = vL
        vR_chosen = vR
        self.paths_to_draw = []
        self.new_positions_to_draw = []

        for vL_possible in vL_possible_array:
            for vR_possible in vR_possible_array:
                # Check velocity limits
                if (vL_possible <= self.robot.max_velocity and
                    vR_possible <= self.robot.max_velocity and
                    vL_possible >= -self.robot.max_velocity and
                    vR_possible >= -self.robot.max_velocity):

                    # Predict position at end of planning horizon
                    x_predict, y_predict, theta_predict, path = \
                        self.robot.predict_position(vL_possible, vR_possible, self.tau)

                    # Store for visualization
                    self.paths_to_draw.append(path)
                    self.new_positions_to_draw.append((x_predict, y_predict))

                    # Calculate trajectory benefit
                    benefit = self._evaluate_trajectory(
                        x, y, x_predict, y_predict, target_x, target_y
                    )

                    # Select best trajectory
                    if benefit > best_benefit:
                        vL_chosen = vL_possible
                        vR_chosen = vR_possible
                        best_benefit = benefit

        # Restore obstacle state
        self.env.barriers = copy.deepcopy(barriers_copy)

        return (vL_chosen, vR_chosen)

    def _evaluate_trajectory(self, x: float, y: float, x_predict: float,
                            y_predict: float, target_x: float,
                            target_y: float) -> float:
        """
        Evaluate trajectory benefit.

        Scoring function:
        benefit = forward_weight * (progress toward goal) - obstacle_weight * (obstacle penalty)

        Args:
            x: Current x position
            y: Current y position
            x_predict: Predicted x position
            y_predict: Predicted y position
            target_x: Target x position
            target_y: Target y position

        Returns:
            Trajectory benefit score (higher is better)
        """
        # 1. Forward progress toward goal
        previous_target_distance = math.sqrt((x - target_x)**2 + (y - target_y)**2)
        new_target_distance = math.sqrt((x_predict - target_x)**2 + (y_predict - target_y)**2)
        distance_forward = previous_target_distance - new_target_distance
        distance_benefit = self.forward_weight * distance_forward

        # 2. Obstacle clearance
        distance_to_obstacle = self.env.calculate_closest_obstacle_distance(
            x_predict, y_predict
        )

        if distance_to_obstacle < self.robot.safe_distance:
            # Penalty increases as we get closer to obstacles
            obstacle_cost = self.obstacle_weight * (self.robot.safe_distance - distance_to_obstacle)
        else:
            obstacle_cost = 0.0

        # Total benefit
        benefit = distance_benefit - obstacle_cost

        return benefit

    def get_visualization_data(self) -> Tuple[List, List]:
        """
        Get trajectory visualization data from last planning cycle.

        Returns:
            Tuple of (paths_to_draw, new_positions_to_draw)
        """
        return (self.paths_to_draw, self.new_positions_to_draw)
