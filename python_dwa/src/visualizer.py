"""
Pygame Visualization

This module handles all visualization including:
- Robot rendering with differential drive wheels
- Obstacle rendering (dynamic and static)
- Trajectory visualization for DWA planning
- GIF recording for demonstrations
"""

import pygame
import math
import time
from PIL import Image
from typing import List, Tuple
from robot import DifferentialDriveRobot
from environment import Environment


class Visualizer:
    """Pygame-based visualization for DWA simulation."""

    def __init__(self, width: int, height: int, scale: float,
                 record_gif: bool = False, gif_duration: float = 15.0,
                 frame_skip: int = 5):
        """
        Initialize visualizer.

        Args:
            width: Screen width (pixels)
            height: Screen height (pixels)
            scale: Pixels per meter conversion factor
            record_gif: Whether to record simulation as GIF
            gif_duration: GIF recording duration (seconds)
            frame_skip: Only capture every Nth frame (reduces file size)
        """
        pygame.init()

        self.width = width
        self.height = height
        self.scale = scale
        self.u0 = width / 2  # Center X
        self.v0 = height / 2  # Center Y

        # Colors
        self.black = (0, 0, 0)
        self.dark_gray = (20, 20, 20)
        self.gray = (70, 70, 70)
        self.white = (255, 255, 255)
        self.light_white = (255, 255, 250)
        self.blue = (0, 0, 255)
        self.light_blue = (0, 120, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.dark_red = (139, 0, 0)

        # Create screen
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dynamic Window Approach - Differential Drive Robot")
        pygame.mouse.set_visible(0)

        # GIF recording
        self.record_gif = record_gif
        self.gif_duration = gif_duration
        self.frame_skip = frame_skip
        self.frames = []
        self.recording = False
        self.start_time = None
        self.frame_count = 0
        self.gif_filename = "../outputs/simulation.gif"

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.

        Args:
            x: World x coordinate (meters)
            y: World y coordinate (meters)

        Returns:
            Tuple of (screen_x, screen_y) in pixels
        """
        u = int(self.u0 + self.scale * x)
        v = int(self.v0 - self.scale * y)
        return (u, v)

    def start_recording(self):
        """Start GIF recording."""
        if self.record_gif and not self.recording:
            self.recording = True
            self.start_time = time.time()
            print("Started recording GIF...")

    def render(self, robot: DifferentialDriveRobot, env: Environment,
               paths: List = None, new_positions: List = None):
        """
        Render complete scene.

        Args:
            robot: Robot instance
            env: Environment instance
            paths: Trajectory paths to visualize (from DWA planning)
            new_positions: Predicted end positions for trajectories
        """
        # Clear screen
        self.screen.fill(self.dark_gray)

        # Draw robot location history (trail)
        for loc in robot.location_history:
            u, v = self.world_to_screen(loc[0], loc[1])
            pygame.draw.circle(self.screen, self.light_white, (u, v), 1, 0)

        # Draw obstacles
        self._draw_obstacles(env)

        # Draw planned trajectories
        if paths:
            self._draw_trajectories(robot, paths)

        # Draw robot
        self._draw_robot(robot)

        # Update display
        pygame.display.flip()

        # Capture frame for GIF
        if self.record_gif and self.recording:
            self._capture_frame()

    def _draw_robot(self, robot: DifferentialDriveRobot):
        """
        Draw robot body and wheels.

        Robot is shown as:
        - White circle for body
        - Two blue circles for differential drive wheels
        """
        x, y, theta = robot.get_pose()
        u, v = self.world_to_screen(x, y)

        # Draw robot body
        pygame.draw.circle(self.screen, self.white,
                         (u, v), int(self.scale * robot.radius), 0)

        # Draw left wheel
        wlx = x - (robot.wheel_width / 2.0) * math.sin(theta)
        wly = y + (robot.wheel_width / 2.0) * math.cos(theta)
        ulx, vlx = self.world_to_screen(wlx, wly)
        wheel_size = 0.04
        pygame.draw.circle(self.screen, self.blue,
                         (ulx, vlx), int(self.scale * wheel_size))

        # Draw right wheel
        wrx = x + (robot.wheel_width / 2.0) * math.sin(theta)
        wry = y - (robot.wheel_width / 2.0) * math.cos(theta)
        urx, vrx = self.world_to_screen(wrx, wry)
        pygame.draw.circle(self.screen, self.blue,
                         (urx, vrx), int(self.scale * wheel_size))

    def _draw_obstacles(self, env: Environment):
        """
        Draw dynamic and static obstacles.

        - Target obstacle: Light blue
        - Other dynamic obstacles: Dark red
        - Static obstacles: Gray (large circles)
        """
        # Dynamic obstacles
        for i, barrier in enumerate(env.barriers):
            if i == env.target_index:
                color = self.light_blue
            else:
                color = self.dark_red

            u, v = self.world_to_screen(barrier[0], barrier[1])
            pygame.draw.circle(self.screen, color,
                             (u, v), int(self.scale * env.barrier_radius), 0)

        # Static obstacles
        for (sx, sy) in env.static_barriers:
            u, v = self.world_to_screen(sx, sy)
            pygame.draw.circle(self.screen, self.gray,
                             (u, v), int(self.scale * 7 * env.barrier_radius), 0)

    def _draw_trajectories(self, robot: DifferentialDriveRobot, paths: List):
        """
        Draw predicted trajectories from DWA planning.

        Paths can be:
        - Type 0: Straight line
        - Type 1: Rotation in place (not drawn)
        - Type 2: Circular arc
        """
        x, y, theta = robot.get_pose()

        for path in paths:
            if path[0] == 0:  # Straight line
                straight_dist = path[1]
                start_u, start_v = self.world_to_screen(x, y)
                end_x = x + straight_dist * math.cos(theta)
                end_y = y + straight_dist * math.sin(theta)
                end_u, end_v = self.world_to_screen(end_x, end_y)
                pygame.draw.line(self.screen, self.green,
                               (start_u, start_v), (end_u, end_v), 1)

            elif path[0] == 2:  # Arc
                cx, cy, R, start_angle, stop_angle = path[1:]

                # Convert arc to screen coordinates
                tlx = int(self.u0 + self.scale * (cx - R))
                tly = int(self.v0 - self.scale * (cy + R))
                rect_width = int(self.scale * (2 * R))
                rect_height = int(self.scale * (2 * R))

                # Normalize angles
                if stop_angle > start_angle:
                    start_a = start_angle
                    stop_a = stop_angle
                else:
                    start_a = stop_angle
                    stop_a = start_angle

                if start_a < 0:
                    start_a += 2 * math.pi
                    stop_a += 2 * math.pi

                # Only draw if valid
                if rect_width > 0 and rect_height > 1 and tlx > 0:
                    try:
                        pygame.draw.arc(self.screen, self.green,
                                      (tlx, tly, rect_width, rect_height),
                                      start_a, stop_a, 1)
                    except:
                        pass  # Skip invalid arcs

    def _capture_frame(self):
        """Capture current frame for GIF recording."""
        elapsed_time = time.time() - self.start_time

        if elapsed_time <= self.gif_duration:
            # Only capture every FRAME_SKIP frames
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                # Capture frame
                frame_surface = pygame.Surface(self.screen.get_size())
                frame_surface.blit(self.screen, (0, 0))

                # Scale down to 50% to reduce file size
                scaled_surface = pygame.transform.scale(
                    frame_surface, (self.width // 2, self.height // 2)
                )

                frame_string = pygame.image.tostring(scaled_surface, 'RGB')
                frame_image = Image.frombytes('RGB',
                                             (self.width // 2, self.height // 2),
                                             frame_string)
                self.frames.append(frame_image)

                # Print progress
                if len(self.frames) % 20 == 0:
                    print(f"Recording: {elapsed_time:.1f}/{self.gif_duration} seconds "
                          f"({len(self.frames)} frames)")
        else:
            # Save GIF and stop recording
            self._save_gif()

    def _save_gif(self):
        """Save recorded frames as GIF."""
        print(f"Saving GIF with {len(self.frames)} frames... Please wait.")
        try:
            self.frames[0].save(
                self.gif_filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=int(self.frame_skip * 100 / 5),  # Adjust timing
                loop=0,
                optimize=False
            )
            print(f"GIF saved successfully as {self.gif_filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
        finally:
            self.recording = False
            self.record_gif = False
            self.frames = []

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            False if quit event received, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
