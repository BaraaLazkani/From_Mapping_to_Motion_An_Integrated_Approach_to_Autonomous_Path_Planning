"""
Visualization utilities for path planning algorithms.

This module provides common visualization functions for drawing robots,
environments, paths, and creating animations.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


def draw_robot(x: float, y: float, theta: float, ax, color: str = 'black', size: int = 50):
    """
    Draw a robot as a circle with an orientation indicator.

    Args:
        x: Robot x-position
        y: Robot y-position
        theta: Robot orientation (radians)
        ax: Matplotlib axis to draw on
        color: Color of the robot marker
        size: Size of the robot marker

    Example:
        >>> fig, ax = plt.subplots()
        >>> draw_robot(5.0, 5.0, math.pi/4, ax)
    """
    # Draw robot body as a circle
    ax.scatter(x, y, color=color, s=size, marker='o', zorder=5)

    # Draw orientation arrow
    dx = 0.5 * math.cos(theta)
    dy = 0.5 * math.sin(theta)
    ax.plot([x, x + dx], [y, y + dy], color=color, linewidth=2, zorder=5)


def draw_environment(ax,
                     centers: List[Tuple[float, float]],
                     sides: List[Tuple[float, float]],
                     path: Optional[List[int]] = None,
                     coordinates: Optional[List[Tuple[float, float]]] = None,
                     start: Optional[Tuple[float, float]] = None,
                     goal: Optional[Tuple[float, float]] = None,
                     path_color: str = 'blue',
                     path_label: str = "Path"):
    """
    Draw the environment with obstacles, start, goal, and optional path.

    Args:
        ax: Matplotlib axis to draw on
        centers: List of obstacle centers [(x1, y1), ...]
        sides: List of obstacle dimensions [(w1, h1), ...]
        path: Optional list of node indices representing the path
        coordinates: Optional list of node coordinates (required if path is provided)
        start: Optional start point (x, y)
        goal: Optional goal point (x, y)
        path_color: Color for the path line
        path_label: Label for the path in legend

    Example:
        >>> fig, ax = plt.subplots()
        >>> centers = [(5, 5), (10, 10)]
        >>> sides = [(2, 2), (3, 3)]
        >>> draw_environment(ax, centers, sides, start=(0, 0), goal=(15, 15))
        >>> plt.show()
    """
    ax.clear()

    # Draw obstacles as grey rectangles
    for i in range(len(centers)):
        cx, cy = centers[i]
        w, h = sides[i]

        # Bottom-left corner of rectangle
        bottom_left = (cx - 0.5 * w, cy - 0.5 * h)

        rectangle = patches.Rectangle(
            bottom_left,
            w, h,
            facecolor='grey',
            edgecolor='black',
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(rectangle)

    # Draw start point (green)
    if start:
        ax.scatter(*start, color='green', s=100, marker='o',
                  label="Start", zorder=10, edgecolors='black', linewidths=1.5)

    # Draw goal point (red)
    if goal:
        ax.scatter(*goal, color='red', s=100, marker='*',
                  label="Goal", zorder=10, edgecolors='black', linewidths=1.5)

    # Draw path if provided
    if path and coordinates:
        path_coords = [coordinates[i] for i in path]
        path_x, path_y = zip(*path_coords)
        ax.plot(path_x, path_y, color=path_color, linewidth=2,
               label=path_label, zorder=3, marker='o', markersize=4)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def setup_plot_limits(ax, x_min: float, x_max: float, y_min: float, y_max: float, margin: float = 1.0):
    """
    Set plot axis limits with optional margin.

    Args:
        ax: Matplotlib axis
        x_min: Minimum x value
        x_max: Maximum x value
        y_min: Minimum y value
        y_max: Maximum y value
        margin: Additional margin around boundaries (default: 1.0)
    """
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)


def save_animation(fig, anim, filename: str, fps: int = 5, writer: str = 'pillow'):
    """
    Save matplotlib animation to file.

    Args:
        fig: Matplotlib figure
        anim: FuncAnimation object
        filename: Output filename (e.g., 'output.gif')
        fps: Frames per second
        writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)

    Example:
        >>> from matplotlib.animation import FuncAnimation
        >>> fig, ax = plt.subplots()
        >>> anim = FuncAnimation(fig, update_func, frames=100)
        >>> save_animation(fig, anim, 'simulation.gif', fps=10)
    """
    anim.save(filename, writer=writer, fps=fps)
    print(f"Animation saved as '{filename}'")
