"""
Gridworld environment for search-based planning agent.
Implements 4-neighbor, unit-cost, deterministic transitions.
"""

from typing import List, Tuple, Optional
import random


#  gridworld for pathfinding experiments
class Gridworld:
    def __init__(self, width: int, height: int, obstacles: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]):
        # Set up grid size, obstacles, start, and goal
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.start = start
        self.goal = goal

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        # Check if position is inside the grid boundaries
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Tuple[int, int]) -> bool:
        # Check if position is not blocked by an obstacle
        return pos not in self.obstacles

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Return all valid 4-neighbor moves (and no diagonals)
        x, y = pos
        candidates = [
            (x - 1, y),  # left
            (x + 1, y),  # right
            (x, y - 1),  # up
            (x, y + 1),  # down
        ]
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def is_goal(self, pos: Tuple[int, int]) -> bool:
        # Check if position is the goal
        return pos == self.goal

    def reset(self) -> Tuple[int, int]:
        # Reset to start position
        return self.start
