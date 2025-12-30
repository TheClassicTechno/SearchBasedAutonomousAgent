from agentic.tools import solve_gridworld_task

# Example task dictionary for a simple 5x5 grid with no obstacles
example_task = {
    "task_id": "demo_001",
    "seed": 42,
    "grid": {
        "width": 5,
        "height": 5,
        "obstacle_density": 0.0,
        "obstacles": [],
        "start": [0, 0],
        "goal": [4, 4],
        "allow_diagonal": False
    },
    "planner": {
        "algorithm": "astar",
        "heuristic": "manhattan",
        "weight": 1.0,
        "max_expansions": 1000,
        "timeout_ms": 1000,
        "tie_break": "lower_h"
    },
    "eval": {
        "compute_oracle_optimal": True,
        "oracle_algorithm": "bfs",
        "record_trace": False
    }
}

if __name__ == "__main__":
    # Call the tool with the correct input format
    result = solve_gridworld_task({"task_json": example_task})
    print("Result:")
    print(result)
