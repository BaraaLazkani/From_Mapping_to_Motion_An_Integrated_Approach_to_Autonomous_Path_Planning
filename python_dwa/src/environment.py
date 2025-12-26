"""
Environment and Obstacle Management

This module handles the dynamic and static obstacles in the environment.
Obstacles move with constant velocity and bounce off boundaries.
"""

import random
import math
from typing import List, Tuple


class Environment:
    """Manages obstacles and environment boundaries."""

    def __init__(self, playfield_bounds: Tuple[float, float, float, float],
                 barrier_radius: float, robot_radius: float):
        """
        Initialize environment.

        Args:
            playfield_bounds: (x_min, y_min, x_max, y_max) in meters
            barrier_radius: Radius of dynamic obstacles (meters)
            robot_radius: Robot radius (meters) for collision checking
        """
        self.bounds = playfield_bounds
        self.barrier_radius = barrier_radius
        self.robot_radius = robot_radius

        # Dynamic obstacles: [x, y, vx, vy]
        self.barriers = []

        # Static large circular obstacles: [(x, y), ...]
        self.static_barriers = []

        # Target obstacle index
        self.target_index = 0

    def add_static_obstacles(self, positions: List[Tuple[float, float]]):
        """
        Add static circular obstacles.

        Args:
            positions: List of (x, y) positions for static obstacles
        """
        self.static_barriers = positions

    def generate_dynamic_obstacles(self, num_obstacles: int, velocity_range: float):
        """
        Generate random dynamic obstacles that avoid static obstacles.

        Args:
            num_obstacles: Number of dynamic obstacles to create
            velocity_range: Maximum obstacle velocity (m/s), Gaussian distribution
        """
        self.barriers = []
        attempts = 0
        max_attempts = num_obstacles * 10

        while len(self.barriers) < num_obstacles and attempts < max_attempts:
            attempts += 1

            # Random position and velocity
            bx = random.uniform(self.bounds[0], self.bounds[2])
            by = random.uniform(self.bounds[1], self.bounds[3])
            vx = random.gauss(0.0, velocity_range)
            vy = random.gauss(0.0, velocity_range)

            # Check if position conflicts with static obstacles
            conflict = False
            for (sx, sy) in self.static_barriers:
                if ((bx <= sx + 7 * self.barrier_radius and bx >= sx - 7 * self.barrier_radius) or
                    (by <= sy + 7 * self.barrier_radius and by >= sy - 7 * self.barrier_radius)):
                    conflict = True
                    break

            if not conflict:
                self.barriers.append([bx, by, vx, vy])

        # Set random target
        if self.barriers:
            self.target_index = random.randint(0, len(self.barriers) - 1)

    def update_obstacles(self, dt: float):
        """
        Update dynamic obstacle positions and handle boundary/obstacle collisions.

        Obstacles bounce off:
        - Playfield boundaries
        - Static obstacles

        Args:
            dt: Time step (seconds)
        """
        for i, barrier in enumerate(self.barriers):
            # Update position
            self.barriers[i][0] += self.barriers[i][2] * dt
            self.barriers[i][1] += self.barriers[i][3] * dt

            # Bounce off playfield boundaries
            if (self.barriers[i][0] < self.bounds[0] or
                self.barriers[i][0] > self.bounds[2]):
                self.barriers[i][2] = -self.barriers[i][2]

            if (self.barriers[i][1] < self.bounds[1] or
                self.barriers[i][1] > self.bounds[3]):
                self.barriers[i][3] = -self.barriers[i][3]

            # Bounce off static obstacles
            for (sx, sy) in self.static_barriers:
                dx = barrier[0] - sx
                dy = barrier[1] - sy
                d = math.sqrt(dx**2 + dy**2)

                # Collision with static obstacle
                if d < 7 * self.barrier_radius:
                    self.barriers[i][2] = -self.barriers[i][2]
                    self.barriers[i][3] = -self.barriers[i][3]

    def calculate_closest_obstacle_distance(self, x: float, y: float) -> float:
        """
        Calculate minimum distance from point (x, y) to any obstacle.

        This is used for collision avoidance in DWA planning.

        Args:
            x: X position (meters)
            y: Y position (meters)

        Returns:
            Closest distance to any obstacle (meters), accounting for robot radius
        """
        closest_dist = 100000.0

        # Check dynamic obstacles (except target)
        for i, barrier in enumerate(self.barriers):
            if i != self.target_index:
                dx = barrier[0] - x
                dy = barrier[1] - y
                d = math.sqrt(dx**2 + dy**2)
                dist = d - self.barrier_radius - self.robot_radius

                if dist < closest_dist:
                    closest_dist = dist

        # Check static obstacles
        for (sx, sy) in self.static_barriers:
            dx = sx - x
            dy = sy - y
            d = math.sqrt(dx**2 + dy**2)
            dist = d - 7 * self.barrier_radius - self.robot_radius

            if dist < closest_dist:
                closest_dist = dist

        return closest_dist

    def check_target_reached(self, x: float, y: float, robot_radius: float) -> bool:
        """
        Check if robot reached the target obstacle.

        Args:
            x: Robot x position (meters)
            y: Robot y position (meters)
            robot_radius: Robot radius (meters)

        Returns:
            True if target reached
        """
        if not self.barriers or self.target_index >= len(self.barriers):
            return False

        target = self.barriers[self.target_index]
        dx = x - target[0]
        dy = y - target[1]
        dist = math.sqrt(dx**2 + dy**2)

        return dist < (self.barrier_radius + robot_radius)

    def set_random_target(self):
        """Set a new random target obstacle."""
        if self.barriers:
            self.target_index = random.randint(0, len(self.barriers) - 1)

    def get_target_position(self) -> Tuple[float, float]:
        """Get current target obstacle position."""
        if self.barriers and self.target_index < len(self.barriers):
            target = self.barriers[self.target_index]
            return (target[0], target[1])
        return (0.0, 0.0)
