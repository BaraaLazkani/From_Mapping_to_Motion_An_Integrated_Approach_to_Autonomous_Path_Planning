"""
LiDAR-based Occupancy Grid Mapping Behavior for Webots Robot.

This module implements a Behavior Tree node that builds a 2D occupancy grid map
using LiDAR sensor data. The map is used by the planning module for path finding.

Behavior Tree Characteristics:
- Status: Returns RUNNING continuously while building the map
- Termination: Triggered by parallel behavior completion
- Output: Saves configuration space map (cspace.npy) and populates blackboard

Coordinate Systems:
- World Frame: Webots simulation coordinates (meters), origin at world center
- Map Frame: Discrete grid (pixels), 200x300 resolution
"""

import py_trees
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def world2map(xw, yw):
    """
    Convert world coordinates to map grid coordinates.

    Coordinate System Details:
    - World Frame: Continuous space in meters
      - X-axis: Horizontal (left/right)
      - Y-axis: Vertical (forward/backward)
      - Origin: Center of simulation world

    - Map Frame: Discrete grid in pixels
      - Dimensions: 200 pixels (width) x 300 pixels (height)
      - Resolution: ~40 pixels/meter (X), ~50 pixels/meter (Y)

    Transformation Equations:
    - px = (xw + 2.25) * 40   → Shifts origin and scales to pixels
    - py = (yw - 2) * (-50)    → Shifts origin, scales, and inverts Y-axis

    Offset Explanation:
    - +2.25 in X: Shifts world origin to align with map grid
    - -2 in Y: Shifts world origin vertically
    - Negative scale in Y: Flips Y-axis (Webots uses +Y forward, map uses +Y down)

    Args:
        xw (float): X-coordinate in world frame (meters)
        yw (float): Y-coordinate in world frame (meters)

    Returns:
        list: [px, py] - Map coordinates in pixels, clamped to valid range

    Example:
        >>> world2map(0.0, 0.0)  # World center
        [90, 100]  # Approximate map center
    """
    # Apply offset and scaling
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))

    # Clamp to valid map bounds (0-199 for X, 0-299 for Y)
    px = min(px, 199)
    py = min(py, 299)
    px = max(px, 0)
    py = max(py, 0)

    return [px, py]


def map2world(px, py):
    """
    Convert map grid coordinates back to world coordinates.

    Inverse transformation of world2map().

    Args:
        px (int): X-coordinate in map frame (pixels)
        py (int): Y-coordinate in map frame (pixels)

    Returns:
        list: [xw, yw] - World coordinates in meters

    Example:
        >>> map2world(90, 100)
        [0.0, 0.0]  # Approximately world center
    """
    xw = px / 40 - 2.25
    yw = py / (-50) + 2
    return [xw, yw]


class Mapping(py_trees.behaviour.Behaviour):
    """
    LiDAR-based occupancy grid mapping behavior.

    This behavior node builds a probabilistic occupancy grid map by:
    1. Reading LiDAR scans at each time step
    2. Transforming scan points from robot frame to world frame
    3. Accumulating evidence of obstacles in a 2D grid
    4. On termination, applying Gaussian blur and thresholding to create C-space

    The mapping runs in parallel with navigation, allowing the robot to
    explore the environment while building the map.

    Attributes:
        hasrun (bool): Whether mapping has started (for termination check)
        robot: Webots Supervisor instance
        blackboard: Shared data storage
        map (np.ndarray): 200x300 occupancy grid (probability values 0-1)
        angles (np.ndarray): LiDAR ray angles (507 valid rays from 667 total)
        gps: GPS sensor device
        compass: Compass sensor device
        lidar: Hokuyo URG-04LX-UG01 LiDAR device
        display: Robot display for visualization
    """

    def __init__(self, name, blackboard):
        """
        Initialize the Mapping behavior.

        Args:
            name (str): Behavior name for debugging
            blackboard: Shared blackboard for data exchange
        """
        super(Mapping, self).__init__(name)
        self.hasrun = False
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard

    def setup(self):
        """
        Initialize Webots devices.

        Called once during behavior tree setup. Acquires references to all
        required sensors and devices.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # Localization sensors
        self.gps = self.robot.getDevice('gps')
        self.compass = self.robot.getDevice('compass')

        # Perception sensor
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')

        # Visualization
        self.display = self.robot.getDevice('display')

        self.logger.debug("  %s [Mapping::setup()]" % self.name)

    def initialise(self):
        """
        Initialize mapping data structures.

        Called once when the behavior first executes. Sets up:
        - Empty 200x300 occupancy grid (all zeros = unknown/free)
        - LiDAR angle array with edge rays filtered out

        LiDAR Configuration:
        - Total rays: 667
        - Field of view: 4.19 radians (~240 degrees)
        - Valid rays: 507 (exclude 80 rays from each end)
        - Why filter edges? Edge rays often have distortion/noise
        """
        self.logger.debug("  %s [Map::initialise()]" % self.name)
        print("mapping!")

        # Create empty occupancy grid (200 pixels wide x 300 pixels tall)
        self.map = np.zeros((200, 300))

        # Generate LiDAR ray angles
        # Full range: 4.19/2 to -4.19/2 radians (symmetric around forward direction)
        self.angles = np.linspace(4.19 / 2, -4.19 / 2, 667)

        # Filter out edge rays (80 from each end) → 507 valid rays remain
        self.angles = self.angles[80:len(self.angles)-80]

    def update(self):
        """
        Execute one mapping iteration.

        This method is called at every simulation timestep while the behavior
        is active. It performs the complete mapping pipeline:

        1. Get robot pose (position + orientation) from GPS and compass
        2. Transform LiDAR scan from robot frame to world frame
        3. Update occupancy grid with new measurements
        4. Visualize trajectory and scan on robot display

        Returns:
            py_trees.common.Status.RUNNING: Always returns RUNNING
                (terminated externally by parallel behavior)
        """
        self.hasrun = True

        # ============================================================
        # Step 1: Get Robot Pose
        # ============================================================
        xw = self.gps.getValues()[0]  # Robot X-position in world frame (meters)
        yw = self.gps.getValues()[1]  # Robot Y-position in world frame (meters)

        # Compute robot orientation from compass
        # Compass returns [x, y] components of north direction vector
        # arctan2(x, y) gives heading angle in radians
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # ============================================================
        # Step 2: Visualize Robot Trajectory
        # ============================================================
        px, py = world2map(xw, yw)
        self.display.setColor(0xFF0000)  # Red color
        self.display.drawPixel(px, py)   # Draw robot's current position

        # ============================================================
        # Step 3: Transform LiDAR Position to World Frame
        # ============================================================
        # LiDAR mounting offset: (0.202, 0, -0.004) meters from robot center
        # We need to account for this offset when transforming scans

        # Build 2D homogeneous transformation matrix: world_T_robot
        # This transforms points from robot frame to world frame
        # Structure: [R | t] where R is 2x2 rotation, t is 2x1 translation
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                          [np.sin(theta),  np.cos(theta), yw],
                          [0, 0, 1]])

        # Transform LiDAR mounting position to world coordinates
        # LiDAR is 0.202m forward of robot center
        lidarPos = w_T_r @ np.array([[0.202], [0], [1]])

        # ============================================================
        # Step 4: Process LiDAR Scan
        # ============================================================
        # Get range measurements (distances to obstacles)
        ranges = np.array(self.lidar.getRangeImage())

        # Replace infinite values (no return) with large distance
        ranges[ranges == np.inf] = 100

        # Filter edge rays (same filtering as angles)
        ranges = ranges[80:len(ranges)-80]  # 507 valid ranges

        # ============================================================
        # Step 5: Transform Scan Points to World Frame
        # ============================================================
        # Rebuild transformation matrix with LiDAR position as origin
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), lidarPos[0][0]],
                          [np.sin(theta),  np.cos(theta), lidarPos[1][0]],
                          [0, 0, 1]])

        # Convert polar coordinates (range, angle) to Cartesian (x, y)
        # In robot frame: x = range * cos(angle), y = range * sin(angle)
        X_i = np.array([ranges * np.cos(self.angles),  # X-coordinates in robot frame
                        ranges * np.sin(self.angles),  # Y-coordinates in robot frame
                        np.ones((507,))])               # Homogeneous coordinates

        # Transform all 507 scan points to world frame
        D = w_T_r @ X_i  # D is 3x507 matrix: [x_world; y_world; 1]

        # ============================================================
        # Step 6: Update Occupancy Grid
        # ============================================================
        # For each detected obstacle point, increment its probability
        for point in D.T:  # Iterate over columns (each point is [x, y, 1])
            px, py = world2map(point[0], point[1])

            # Accumulate evidence (0.001 per detection)
            # Multiple scans of same obstacle increase confidence
            self.map[px, py] += 0.001

            # Visualize on display (grayscale intensity = probability)
            v = int(min(self.map[px, py], 1.0) * 255)  # Clamp to [0, 255]
            color = (v * 256**2 + v * 256 + v)  # Convert to RGB (grayscale)
            self.display.setColor(color)
            self.display.drawPixel(px, py)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Finalize map and save configuration space.

        Called when the behavior is terminated (parallel navigation completes).
        This method:
        1. Applies Gaussian blur to smooth occupancy probabilities
        2. Thresholds to create binary configuration space (C-space)
        3. Saves C-space to disk for future use
        4. Populates blackboard for immediate path planning
        5. Visualizes final map on robot display

        Configuration Space (C-space):
        - Binary map where 1 = obstacle, 0 = free space
        - Includes safety margin around obstacles (from Gaussian blur)
        - Used by A* planner to find collision-free paths

        Gaussian Blur Rationale:
        - Kernel size: 30x30 pixels → ~0.75m safety margin
        - Smooths noisy measurements
        - Enlarges obstacles to account for robot size
        - Creates gradient for conservative planning

        Threshold (0.9):
        - Probability > 0.9 → mark as obstacle
        - Conservative threshold reduces false negatives
        - Ensures robot maintains clearance from walls

        Args:
            new_status: New behavior status (SUCCESS/FAILURE/INVALID)
        """
        self.logger.debug("  %s [Mapping::terminate()][%s->%s]" %
                         (self.name, self.status, new_status))

        if self.hasrun:
            # ============================================================
            # Step 1: Apply Gaussian Blur (Obstacle Dilation)
            # ============================================================
            # Convolve with 30x30 uniform kernel → equivalent to Gaussian blur
            # This spreads obstacle probability to neighboring cells
            # Result: Obstacles appear larger = safety margin for robot
            cmap = signal.convolve2d(self.map, np.ones((30, 30)), mode='same')

            # ============================================================
            # Step 2: Threshold to Create Binary C-space
            # ============================================================
            # Convert continuous probabilities to binary obstacle map
            # Threshold at 0.9: high confidence required to mark as obstacle
            cspace = cmap > 0.9

            # ============================================================
            # Step 3: Save to Disk
            # ============================================================
            np.save('cspace', cspace)  # Save as cspace.npy
            map = np.load('cspace.npy')  # Reload to verify

            # ============================================================
            # Step 4: Visualize Final Map (Optional)
            # ============================================================
            plt.figure(0)
            plt.imshow(map)
            plt.show()

            # ============================================================
            # Step 5: Populate Blackboard for Path Planning
            # ============================================================
            self.blackboard.write('map', map)

            # ============================================================
            # Step 6: Display Final Map on Robot Screen
            # ============================================================
            # Draw all obstacle cells in white on robot display
            for px in range(len(map)):
                for py in range(len(map[0])):
                    if map[px][py] > 0:
                        self.display.setColor(0xFFFFFF)  # White
                        self.display.drawPixel(px, py)
