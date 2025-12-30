"""
Random obstacle generator for Gridworld.
"""

import random
from typing import List, Tuple


# Generate random obstacles, avoiding start and goal
def generate_obstacles(width: int, height: int, density: float, seed: int, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Set random seed for reproducibility
    random.seed(seed)
    total_cells = width * height
    num_obstacles = int(total_cells * density)
    # Exclude start and goal from obstacles
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) != start and (x, y) != goal]
    obstacles = random.sample(all_cells, min(num_obstacles, len(all_cells)))
    return obstacles
