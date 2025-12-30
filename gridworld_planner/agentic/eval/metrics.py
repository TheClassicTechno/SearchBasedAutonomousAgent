"""
Metrics for evaluating planning runs.
"""

def optimality_gap(path_cost, optimal_cost):

    # Return None if the optimal cost is not provided or zero (avoid divide by zero)
    if optimal_cost is None or optimal_cost == 0:
        return None
    # Compute the ratio of the agent's path cost to the optimal path cost
    return path_cost / optimal_cost
