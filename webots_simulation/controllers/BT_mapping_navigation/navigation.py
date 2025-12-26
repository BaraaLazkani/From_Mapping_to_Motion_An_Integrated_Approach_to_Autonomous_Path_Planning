"""
Waypoint Following Navigation Behavior for Webots Robot.

This module implements a Behavior Tree node that executes motor commands to follow
a sequence of waypoints using a PD-like control law for differential drive robots.

Behavior Tree Characteristics:
- Status: Returns RUNNING while navigating, SUCCESS when all waypoints reached
- Execution: Runs continuously after path planning completes
- Input: Reads waypoint list from blackboard (populated by planning behavior)
- Output: Sets motor velocities to navigate along the path

Control Strategy:
- Differential Drive Control: Independent left/right wheel velocities
- PD-like Control Law: Combines distance and heading error
- Waypoint Switching: Advances to next waypoint when within threshold

Robot Model:
- Type: Differential drive (TurtleBot3 Burger)
- Actuators: Two independent wheel motors
- Max velocity: 6.28 rad/s (2π rad/s = 1 revolution/second)
- Wheel radius: ~0.033m → max linear velocity ~0.21 m/s

Control Variables:
- rho (ρ): Euclidean distance to current waypoint (meters)
- alpha (α): Heading error - angle from robot heading to waypoint (radians)
- theta (θ): Robot orientation from compass (radians)
"""

import py_trees
import numpy as np


class Navigation(py_trees.behaviour.Behaviour):
    """
    Waypoint following navigation behavior using differential drive control.

    This behavior implements a PD-like controller that navigates the robot through
    a sequence of waypoints. The control law computes individual wheel velocities
    based on distance to waypoint (ρ) and heading error (α).

    Control Law Derivation:
    For a differential drive robot, we want:
    1. Turn toward waypoint → reduce heading error α
    2. Drive forward → reduce distance ρ

    Differential drive velocities:
    - vL = left wheel velocity
    - vR = right wheel velocity

    Control equations:
    - vL = -p1·α + p2·ρ  (turn left when α > 0, drive forward based on ρ)
    - vR = +p1·α + p2·ρ  (turn right when α > 0, drive forward based on ρ)

    Gains:
    - p1 = 4: Proportional gain for heading correction
      - Higher p1 → faster turning response
      - Too high → oscillations
    - p2 = 2: Proportional gain for forward motion
      - Higher p2 → faster approach to waypoint
      - Ratio p1/p2 = 2 balances turning vs driving

    Velocity Saturation:
    - Commanded velocities clamped to [-6.28, 6.28] rad/s
    - 6.28 ≈ 2π rad/s = 1 full wheel revolution per second
    - Prevents exceeding motor limits and ensures stability

    Waypoint Switching:
    - Advance to next waypoint when ρ < 0.4 meters
    - Threshold chosen to balance accuracy vs smooth motion
    - Too small → robot oscillates around waypoint
    - Too large → cuts corners

    Attributes:
        robot: Webots Supervisor instance
        blackboard: Shared data storage
        WP (list): Waypoint sequence in world coordinates
        index (int): Current waypoint index
        gps: GPS sensor for position feedback
        compass: Compass sensor for orientation feedback
        leftMotor: Left wheel motor actuator
        rightMotor: Right wheel motor actuator
        marker: Visual marker showing current target waypoint
    """

    def __init__(self, name, blackboard):
        """
        Initialize the Navigation behavior.

        Args:
            name (str): Behavior name for debugging
            blackboard: Shared blackboard for data exchange
        """
        super(Navigation, self).__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard

    def setup(self):
        """
        Initialize Webots devices.

        Called once during behavior tree setup. Acquires references to all
        required sensors and actuators.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # Localization sensors
        self.gps = self.robot.getDevice('gps')
        self.compass = self.robot.getDevice('compass')

        # Motor actuators
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')

        # Visual marker for current target waypoint
        self.marker = self.robot.getFromDef("marker").getField("translation")

        self.logger.debug("  %s [Navigation::setup()]" % self.name)

    def initialise(self):
        """
        Initialize navigation state.

        Called once when the behavior first executes. Stops motors and reads
        the waypoint sequence from the blackboard.
        """
        print("navigating!")

        # Ensure robot starts from rest
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        # Start at first waypoint
        self.index = 0

        self.logger.debug("  %s [Navigation::initialise()]" % self.name)

        # Read waypoint sequence from blackboard (written by planning behavior)
        self.WP = self.blackboard.read('waypoints')

    def update(self):
        """
        Execute one navigation control iteration.

        This method is called at every simulation timestep while the behavior
        is active. It performs the complete navigation control loop:

        1. Get robot pose (position + orientation) from sensors
        2. Compute control variables (ρ, α) relative to current waypoint
        3. Apply PD control law to compute wheel velocities
        4. Saturate velocities to motor limits
        5. Check if waypoint reached and advance if necessary

        Control Loop Breakdown:

        Step 1: State Estimation
        - Read GPS → (xw, yw) robot position in world frame
        - Read compass → θ robot heading angle

        Step 2: Compute Control Variables
        - ρ = distance to current waypoint
        - α = heading error (angle from robot heading to waypoint)

        Step 3: Apply Control Law
        - vL = -p1·α + p2·ρ
        - vR = +p1·α + p2·ρ

        Step 4: Velocity Saturation
        - Clamp to [-6.28, 6.28] rad/s to respect motor limits

        Step 5: Waypoint Switching
        - If ρ < 0.4m, advance to next waypoint
        - If last waypoint reached, return SUCCESS

        Returns:
            py_trees.common.Status.RUNNING: Navigation in progress
            py_trees.common.Status.SUCCESS: All waypoints reached
        """
        self.logger.debug("  %s [Navigation::update()]" % self.name)

        # ============================================================
        # Step 1: Get Robot Pose
        # ============================================================
        # Robot position in world frame (meters)
        xw = self.gps.getValues()[0]  # X-coordinate
        yw = self.gps.getValues()[1]  # Y-coordinate

        # Robot orientation (radians)
        # Compass returns [x, y] components of north direction vector
        # arctan2(x, y) gives heading angle relative to north
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # ============================================================
        # Step 2: Compute Control Variables
        # ============================================================
        # Distance to current waypoint (ρ)
        # ρ = √[(xw - x_goal)² + (yw - y_goal)²]
        rho = np.sqrt((xw - self.WP[self.index][0])**2 + (yw - self.WP[self.index][1])**2)

        # Heading error (α)
        # α = atan2(y_goal - yw, x_goal - xw) - θ
        # Positive α: waypoint is to the left → robot should turn left
        # Negative α: waypoint is to the right → robot should turn right
        alpha = np.arctan2(self.WP[self.index][1] - yw, self.WP[self.index][0] - xw) - theta

        # Normalize angle to [-π, π]
        # This ensures shortest rotation to waypoint
        if (alpha > np.pi):
            alpha = alpha - 2 * np.pi
        elif (alpha < -np.pi):
            alpha = alpha + 2 * np.pi

        # ============================================================
        # Step 3: Visualize Current Target Waypoint
        # ============================================================
        # Update marker position in simulation for visualization
        self.marker.setSFVec3f([*self.WP[self.index], 0])

        # ============================================================
        # Step 4: Apply PD Control Law
        # ============================================================
        # Control gains
        p1 = 4  # Heading correction gain (proportional to α)
        p2 = 2  # Forward motion gain (proportional to ρ)

        # Differential drive control equations:
        #
        # Left wheel velocity:
        #   vL = -p1·α + p2·ρ
        #   - When α > 0 (waypoint on left): vL decreases → robot turns left
        #   - When α < 0 (waypoint on right): vL increases → robot turns right
        #   - ρ term drives both wheels forward based on distance
        #
        # Right wheel velocity:
        #   vR = +p1·α + p2·ρ
        #   - When α > 0 (waypoint on left): vR increases → robot turns left
        #   - When α < 0 (waypoint on right): vR decreases → robot turns right
        #   - ρ term drives both wheels forward based on distance
        #
        # Turning behavior:
        #   - vR - vL = 2·p1·α → differential velocity proportional to heading error
        #   - vR + vL = 2·p2·ρ → average velocity proportional to distance
        vL = -p1 * alpha + p2 * rho
        vR = +p1 * alpha + p2 * rho

        # ============================================================
        # Step 5: Velocity Saturation
        # ============================================================
        # Clamp velocities to motor limits: [-6.28, 6.28] rad/s
        # 6.28 ≈ 2π rad/s = 1 revolution/second
        # This prevents:
        #   1. Exceeding motor capabilities
        #   2. Instability from excessive gains
        #   3. Wheel slip from rapid acceleration
        vL = min(vL, 6.28)
        vR = min(vR, 6.28)
        vL = max(vL, -6.28)
        vR = max(vR, -6.28)

        # ============================================================
        # Step 6: Send Motor Commands
        # ============================================================
        self.leftMotor.setVelocity(vL)
        self.rightMotor.setVelocity(vR)

        # ============================================================
        # Step 7: Waypoint Switching Logic
        # ============================================================
        # Check if current waypoint reached
        if (rho < 0.4):
            # Within 0.4 meters → advance to next waypoint
            self.index = self.index + 1

            # Check if all waypoints completed
            if self.index == len(self.WP):
                self.feedback_message = "Last waypoint reached"
                return py_trees.common.Status.SUCCESS

        # Still navigating
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Clean up navigation state on termination.

        Called when the behavior is terminated (either SUCCESS after reaching goal,
        or FAILURE/INVALID if interrupted). Ensures robot stops and resets state.

        Args:
            new_status: New behavior status (SUCCESS/FAILURE/INVALID)
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

        # Reset waypoint index to 0 for potential re-execution
        self.index = 0

        # Stop robot motors
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
