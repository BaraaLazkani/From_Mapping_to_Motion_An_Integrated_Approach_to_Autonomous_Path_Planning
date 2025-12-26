"""
Cubic Spline Curve (CSC) trajectory smoothing.

This module implements smooth trajectory generation between waypoints
using cubic spline curves with oscillating approach and linear motion phases.
"""

import math
import numpy as np


class CSCTrajectory:
    """
    Cubic Spline Curve trajectory generator for smooth robot motion.

    This class generates smooth trajectories between waypoints by dividing
    the motion into three phases:
    1. Oscillating approach to intermediate point (0 ≤ t ≤ 10)
    2. Linear motion toward final waypoint (10 < t ≤ 20)
    3. Settling at final position (20 < t ≤ 30)

    The trajectory ensures continuous position and orientation.

    Attributes:
        r (float): Robot turning radius
        b (float): Robot width
        omg (float): Angular frequency (2π/10)
        cnt (int): Counter for trajectory segments
        temp: Storage for previous z2_20 value for continuity
    """

    def __init__(self, R: float, r_width: float):
        """
        Initialize the CSC trajectory generator.

        Args:
            R: Robot turning radius (meters)
            r_width: Robot width (meters)

        Example:
            >>> trajectory = CSCTrajectory(R=1.0, r_width=0.5)
        """
        self.r = R
        self.b = r_width
        self.omg = 2 * math.pi / 10  # Angular frequency
        self.cnt = 0
        self.temp = None

    def set_cond(self,
                 initial_cond: Tuple[float, float, float],
                 final_cond: Tuple[float, float, float]):
        """
        Set initial and final conditions for trajectory segment.

        This method computes the cubic spline coefficients (c1, c2, c3)
        and intermediate waypoints for smooth motion.

        Args:
            initial_cond: (x0, y0, theta0) - initial position and orientation
            final_cond: (x_f, y_f, theta_f) - final position and orientation

        Mathematical Model:
            The trajectory is parameterized by:
            - z1: x-position
            - z2: tan(orientation)
            - z3: y-position

            Coefficients are solved to ensure continuity of position and
            orientation at intermediate points.

        Example:
            >>> traj = CSCTrajectory(1.0, 0.5)
            >>> traj.set_cond((0, 0, 0), (10, 10, math.pi/4))
        """
        # Extract initial conditions
        self.z1_0 = initial_cond[0]

        # Use previous segment's final orientation for continuity
        if self.cnt == 0:
            self.z2_0 = math.tan(initial_cond[2])
        else:
            self.z2_0 = self.temp

        self.z3_0 = initial_cond[1]

        # Extract final conditions
        self.z1_30 = final_cond[0]
        self.z2_30 = math.tan(final_cond[2])
        self.z3_30 = final_cond[1]

        # Prevent division by zero
        if self.z1_0 == self.z1_30:
            self.z1_30 += 0.1

        # Compute cubic spline coefficients
        self.c2 = (self.z1_30 - self.z1_0) / 10
        self.c1 = (self.z3_30 - self.z3_0) / (100 * self.c2) - self.z2_0 / 10
        self.c3 = (self.z2_30 - 10 * self.c1 - self.z2_0) / 10

        # Intermediate waypoints (at t=10 and t=20)
        self.z1_10 = self.z1_0
        self.z2_10 = 10 * self.c1 + self.z2_0
        self.z3_10 = self.z3_0

        self.z1_20 = 10 * self.c2 + self.z1_10
        self.z2_20 = self.z2_10
        self.z3_20 = 10 * self.c2 * self.z2_10 + self.z3_10

    def update_pos(self, t: float) -> Tuple[float, float, float]:
        """
        Compute robot position and orientation at time t.

        The trajectory has three phases:
        - Phase 1 (0 ≤ t ≤ 10): Oscillating approach with changing orientation
        - Phase 2 (10 < t ≤ 20): Linear motion with constant orientation
        - Phase 3 (20 < t ≤ 30): Settling at final position

        Args:
            t: Time parameter (0 to 30)

        Returns:
            Tuple of (x, y, orientation_angle) at time t

        Mathematical Formulation:
            Phase 1: Uses oscillating functions with sin(ωt) terms
            Phase 2: Linear interpolation
            Phase 3: Holds final position

        Example:
            >>> traj = CSCTrajectory(1.0, 0.5)
            >>> traj.set_cond((0, 0, 0), (10, 10, 0))
            >>> x, y, theta = traj.update_pos(5.0)  # Midpoint of phase 1
        """
        # Phase 1: Oscillating approach (0 ≤ t ≤ 10)
        if t >= 0 and t <= 10:
            x = self.z1_0
            y = self.z3_0
            orientation_angle = math.atan(
                self.c1 * t - (self.c1 / self.omg) * math.sin(self.omg * t) + self.z2_0
            )

        # Phase 2: Linear motion (10 < t ≤ 20)
        elif t > 10 and t <= 20:
            x = (self.c2 * t - (self.c2 / self.omg) * math.sin(self.omg * t) +
                 self.z1_10 - 10 * self.c2)
            y = (self.z2_10 * self.c2 * t - (self.z2_10 * self.c2 / self.omg) *
                 math.sin(self.omg * t) + self.z3_10 - 10 * self.z2_10 * self.c2)
            orientation_angle = math.atan(self.z2_10)

        # Phase 3: Settling (20 < t ≤ 30)
        else:  # t > 20
            x = self.z1_30
            y = self.z3_30
            orientation_angle = math.atan(self.z2_10)

        # Update state for next segment
        self.cnt += 1
        self.temp = self.z2_20

        return x, y, orientation_angle
