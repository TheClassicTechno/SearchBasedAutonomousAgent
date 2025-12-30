"""
Utility functions for search algorithms.
"""

def reconstruct_path(came_from, start, goal):
    # Reconstruct the path from goal to start using parent links
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            # If a node is missing, the path is incomplete
            return None
    path.append(start)
    path.reverse()
    return path
