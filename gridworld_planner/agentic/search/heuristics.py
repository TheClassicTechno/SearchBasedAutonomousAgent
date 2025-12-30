"""
Heuristic functions for Gridworld planning.
"""

def manhattan(pos, goal):
    # Compute Manhattan distance between two points (and no diagonals)
    x, y = pos
    gx, gy = goal
    return abs(x - gx) + abs(y - gy)

def weighted_manhattan(pos, goal, weight=1.0):
    # Weighted Manhattan distance for weighted A* search
    return weight * manhattan(pos, goal)
