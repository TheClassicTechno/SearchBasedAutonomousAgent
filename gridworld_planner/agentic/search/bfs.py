"""
Breadth-First Search (BFS) for Gridworld (unit-cost, 4-neighbor).
"""

from collections import deque
from typing import Tuple, List, Dict, Optional
from agentic.env.gridworld import Gridworld


# Simple BFS for shortest path in grid
def bfs_search(env: Gridworld, max_expansions=200000):
    # Initialize search structures
    start = env.start
    goal = env.goal
    frontier = deque([start])  # FIFO queue for open nodes
    came_from = {start: None}  # track parent links for path
    nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        # Pop node from front of queue
        current = frontier.popleft()
        nodes_expanded += 1
        if env.is_goal(current):
            break
        for neighbor in env.neighbors(current):
            # Only visit each node once
            if neighbor not in came_from:
                frontier.append(neighbor)
                came_from[neighbor] = current
        max_frontier_size = max(max_frontier_size, len(frontier))
        if nodes_expanded >= max_expansions:
            # Stop if expansion budget exceeded
            return None, nodes_expanded, max_frontier_size, 'budget_exceeded'
    else:
        # No path found
        return None, nodes_expanded, max_frontier_size, 'no_path'

    # Reconstruct path from goal to start
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return None, nodes_expanded, max_frontier_size, 'no_path'
    path.append(start)
    path.reverse()
    return path, nodes_expanded, max_frontier_size, 'goal_reached'
