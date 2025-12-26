"""
Map Persistence Behavior for Webots Robot.

This module implements a Behavior Tree node that checks if a pre-existing map
is available on disk. If found, it loads and populates the map data, allowing
the robot to skip the time-consuming mapping phase.

Behavior Tree Characteristics:
- Status: Returns SUCCESS if map exists, FAILURE if not
- Execution: Runs once at the start of the behavior tree
- Purpose: Enable map reuse across multiple simulation runs
- Output: Loads map into blackboard and displays on robot screen

Map Persistence:
- File: cspace.npy (NumPy binary format)
- Content: Binary occupancy grid (200x300)
  - 0: Free space
  - 1: Obstacle (after thresholding)
- Location: Current working directory (controller folder)

Why Map Persistence?
1. **Efficiency**: Mapping takes ~60 seconds, loading takes <1 second
2. **Repeatability**: Same map for multiple planning experiments
3. **Development**: Test planning/navigation without re-mapping
4. **Robustness**: Handle simulation restarts gracefully

Behavior Tree Integration:
- Used in Selector node: [DoesMapExist, Mapping]
- If DoesMapExist returns SUCCESS → Selector succeeds, skip Mapping
- If DoesMapExist returns FAILURE → Selector tries Mapping behavior
"""

from os.path import exists
import py_trees
import numpy as np


class DoesMapExist(py_trees.behaviour.Behaviour):
    """
    Map existence check and loading behavior.

    This behavior implements a file-based map persistence mechanism. It checks
    if a configuration space map has been previously saved to disk. If found,
    the map is loaded into the blackboard and visualized on the robot's display.

    File Format:
    - Filename: cspace.npy
    - Format: NumPy binary (.npy)
    - Data type: Boolean or integer array
    - Dimensions: 200x300 (width x height)

    Map Processing:
    - Loaded values are binary (0 or 1)
    - Values > 0 are scaled to 255 for visualization consistency
    - This matches the grayscale intensity used in mapping.py

    Display Visualization:
    - Obstacles (value > 0) → white pixels (0xFFFFFF)
    - Free space (value = 0) → black pixels (default)
    - Provides immediate visual feedback that map loaded correctly

    Attributes:
        robot: Webots Supervisor instance
        blackboard: Shared data storage
        display: Robot display device for visualization
    """

    def __init__(self, name, blackboard):
        """
        Initialize the DoesMapExist behavior.

        Args:
            name (str): Behavior name for debugging
            blackboard: Shared blackboard for data exchange
        """
        super(DoesMapExist, self).__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard

    def setup(self):
        """
        Initialize Webots devices.

        Called once during behavior tree setup. Acquires reference to the
        robot display for map visualization.
        """
        self.display = self.robot.getDevice('display')
        self.logger.debug("  %s [Mapping::setup()]" % self.name)

    def update(self):
        """
        Check if map file exists and load if available.

        This method performs the complete map loading pipeline:

        1. Check if cspace.npy exists in current directory
        2. If exists:
           - Load map from disk using numpy
           - Scale obstacle values to 255 for consistency
           - Write map to blackboard for planning behavior
           - Visualize map on robot display
           - Return SUCCESS (signals Selector to skip mapping)
        3. If not exists:
           - Return FAILURE (signals Selector to proceed with mapping)

        File Location:
        - Current working directory is the controller folder
        - Path: controllers/BT_mapping_navigation/cspace.npy

        Returns:
            py_trees.common.Status.SUCCESS: Map loaded successfully
            py_trees.common.Status.FAILURE: Map file not found
        """
        # ============================================================
        # Step 1: Check File Existence
        # ============================================================
        # exists() checks current working directory (controller folder)
        file_exists = exists('cspace.npy')

        if file_exists:
            print("Map already exists")

            # ========================================================
            # Step 2: Load Map from Disk
            # ========================================================
            # Load binary occupancy grid from NumPy file
            # Expected shape: (200, 300)
            # Expected values: 0 (free) or 1 (obstacle)
            map = np.load('cspace.npy')

            # ========================================================
            # Step 3: Scale Obstacle Values
            # ========================================================
            # Scale obstacle cells (value > 0) to 255
            # Why 255?
            # - Matches grayscale intensity convention (0-255)
            # - Consistent with visualization in mapping.py
            # - Makes obstacles clearly visible on display
            map[map > 0] = 255

            # ========================================================
            # Step 4: Populate Blackboard
            # ========================================================
            # Write loaded map to blackboard for planning behavior
            # Planning behavior expects 'map' key with binary grid
            self.blackboard.write('map', map)

            # ========================================================
            # Step 5: Visualize on Robot Display
            # ========================================================
            # Draw all obstacle cells in white on robot display
            # This provides visual confirmation that map loaded correctly
            for px in range(len(map)):
                for py in range(len(map[0])):
                    if map[px][py] > 0:
                        self.display.setColor(0xFFFFFF)  # White
                        self.display.drawPixel(px, py)

            # Map loaded successfully → Selector succeeds, skip mapping
            return py_trees.common.Status.SUCCESS
        else:
            print("Map does not exist")

            # Map not found → Selector fails, proceed to mapping behavior
            return py_trees.common.Status.FAILURE
