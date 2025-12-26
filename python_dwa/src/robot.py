"""
Robot Model and Kinematics

This module defines the differential drive robot model and its kinematic equations.
The robot has two wheels separated by distance W, each with independent velocities.

Kinematics:
- Straight motion: vL = vR
- Pure rotation: vL = -vR
- Arc motion: General case with turning radius R
"""

import math
from typing import Tuple


class DifferentialDriveRobot:
    """Differential drive robot with kinematic motion model."""

    def __init__(self, radius: float, wheel_width: float, max_velocity: float,
                 max_acceleration: float, safe_distance: float):
        """
        Initialize robot parameters.

        Args:
            radius: Robot radius (meters)
            wheel_width: Distance between wheels (meters)
            max_velocity: Maximum wheel velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            safe_distance: Minimum safe distance from obstacles (meters)
        """
        self.radius = radius
        self.wheel_width = wheel_width
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.safe_distance = safe_distance

        # Current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Current velocities
        self.vL = 0.0
        self.vR = 0.0

        # Position history for visualization
        self.location_history = []

    def set_pose(self, x: float, y: float, theta: float):
        """Set robot pose."""
        self.x = x
        self.y = y
        self.theta = theta

    def set_velocities(self, vL: float, vR: float):
        """Set wheel velocities."""
        self.vL = vL
        self.vR = vR

    def predict_position(self, vL: float, vR: float, deltat: float) -> Tuple[float, float, float, tuple]:
        """
        Predict robot position after time deltat with given wheel velocities.

        This implements the differential drive kinematic model:
        - Straight: Both wheels same speed → linear motion
        - Rotation: Wheels opposite speeds → rotate in place
        - Arc: Different wheel speeds → circular arc motion

        Args:
            vL: Left wheel velocity (m/s)
            vR: Right wheel velocity (m/s)
            deltat: Time step (seconds)

        Returns:
            Tuple of (x_new, y_new, theta_new, path_info)
            path_info is used for visualization
        """
        # Straight motion: vL ≈ vR
        if round(vL, 3) == round(vR, 3):
            xnew = self.x + vL * deltat * math.cos(self.theta)
            ynew = self.y + vL * deltat * math.sin(self.theta)
            thetanew = self.theta
            path = (0, vL * deltat)  # Type 0: straight line

        # Pure rotation: vL ≈ -vR
        elif round(vL, 3) == -round(vR, 3):
            xnew = self.x
            ynew = self.y
            thetanew = self.theta + ((vR - vL) * deltat / self.wheel_width)
            path = (1, 0)  # Type 1: rotation in place

        # Arc motion: General case
        else:
            # Turning radius: R = (W/2) * (vR + vL) / (vR - vL)
            R = self.wheel_width / 2.0 * (vR + vL) / (vR - vL)
            deltatheta = (vR - vL) * deltat / self.wheel_width

            # New position using ICC (Instantaneous Center of Curvature)
            xnew = self.x + R * (math.sin(deltatheta + self.theta) - math.sin(self.theta))
            ynew = self.y - R * (math.cos(deltatheta + self.theta) - math.cos(self.theta))
            thetanew = self.theta + deltatheta

            # Arc visualization info (center, radius, angles)
            cx = self.x - R * math.sin(self.theta)
            cy = self.y + R * math.cos(self.theta)
            Rabs = abs(R)

            if R > 0:
                start_angle = self.theta - math.pi / 2.0
            else:
                start_angle = self.theta + math.pi / 2.0
            stop_angle = start_angle + deltatheta

            path = (2, cx, cy, Rabs, start_angle, stop_angle)  # Type 2: arc

        return (xnew, ynew, thetanew, path)

    def update(self, vL: float, vR: float, dt: float):
        """
        Update robot state with new velocities.

        Args:
            vL: Left wheel velocity (m/s)
            vR: Right wheel velocity (m/s)
            dt: Time step (seconds)
        """
        self.location_history.append((self.x, self.y))
        (self.x, self.y, self.theta, _) = self.predict_position(vL, vR, dt)
        self.vL = vL
        self.vR = vR

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose."""
        return (self.x, self.y, self.theta)

    def get_velocities(self) -> Tuple[float, float]:
        """Get current wheel velocities."""
        return (self.vL, self.vR)
