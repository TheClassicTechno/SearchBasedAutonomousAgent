"""
A* search for Gridworld (unit-cost, 4-neighbor)
"""

import heapq
from typing import Tuple, List, Dict, Optional
from agentic.env.gridworld import Gridworld
from agentic.search.heuristics import manhattan, weighted_manhattan


# A* search for shortest path in grid
def astar_search(env: Gridworld, heuristic_fn, weight=1.0, max_expansions=200000, timeout_ms=None):
    # Initialize search structures
    start = env.start
    goal = env.goal
    frontier = []  # priority queue for open nodes
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}  # track parent links for path
    cost_so_far = {start: 0}  # g cost for each node
    nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        # Pop node with lowest f = g + h
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1
        if env.is_goal(current):
            break
        for neighbor in env.neighbors(current):
            new_cost = cost_so_far[current] + 1
            # Only update if new path is better
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = heuristic_fn(neighbor, goal) if weight == 1.0 else weighted_manhattan(neighbor, goal, weight)
                priority = new_cost + h
                heapq.heappush(frontier, (priority, neighbor))
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
            # If a node is missing, path is incomplete
            return None, nodes_expanded, max_frontier_size, 'no_path'
    path.append(start)
    path.reverse()
    return path, nodes_expanded, max_frontier_size, 'goal_reached'
