"""
Batch evaluation script for running ablations over Gridworld tasks.
"""
import os
import json
from agentic.eval.runner import run_task_from_dict
from agentic.logging_utils import JsonlLogger
from agentic.env.generators import generate_obstacles


def make_task(task_id, width, height, density, seed, algorithm, heuristic, weight=1.0):
    # Create a single planning task specification for batch evaluation
    start = (0, 0)
    goal = (width - 1, height - 1)
    obstacles = generate_obstacles(width, height, density, seed, start, goal)
    return {
        "task_id": task_id,
        "seed": seed,
        "grid": {
            "width": width,
            "height": height,
            "obstacle_density": density,
            "obstacles": obstacles,
            "start": list(start),
            "goal": list(goal),
            "allow_diagonal": False
        },
        "planner": {
            "algorithm": algorithm,
            "heuristic": heuristic,
            "weight": weight,
            "max_expansions": 200000,
            "timeout_ms": 2000,
            "tie_break": "lower_h"
        },
        "eval": {
            "compute_oracle_optimal": True,
            "oracle_algorithm": "bfs",
            "record_trace": False
        }
    }

def run_batch():
    # Run a batch of planning tasks across grid sizes, densities, algorithms, and seeds
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "..", "..", "results", f"exp_2025-12-29_01")
    out_dir = os.path.abspath(out_dir)
    logger = JsonlLogger(out_dir)
    grid_sizes = [10, 20, 30, 40, 50]
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    algorithms = ["bfs", "astar", "mcts"]
    heuristics = ["manhattan", "weighted"]
    # weighted = A weighted version of the Manhattan heuristic
    # makes A* more aggressive, reducing search time at the cost of solution optimality
    weights = [1.0, 1.5, 2.0]
    seeds = [42, 43, 44]
    task_id = 0
    # Iterate over all combinations of parameters
    for width in grid_sizes:
        for density in densities:
            for seed in seeds:
                for algorithm in algorithms:
                    if algorithm == "bfs":
                        # BFS does not use weighted heuristics
                        task = make_task(f"gw_{task_id:06d}", width, width, density, seed, algorithm, "manhattan", 1.0)
                        result = run_task_from_dict(task)
                        logger.log_run(result)
                        task_id += 1
                    elif algorithm == "astar":
                        for weight in weights:
                            heuristic = "manhattan" if weight == 1.0 else "weighted"
                            task = make_task(f"gw_{task_id:06d}", width, width, density, seed, algorithm, heuristic, weight)
                            result = run_task_from_dict(task)
                            logger.log_run(result)
                            task_id += 1
                    elif algorithm == "mcts":
                        # MCTS does not use heuristics or weights, so pass defaults
                        task = make_task(f"gw_{task_id:06d}", width, width, density, seed, algorithm, "none", 1.0)
                        result = run_task_from_dict(task)
                        logger.log_run(result)
                        task_id += 1
    print(f"Results written to: {out_dir}")

if __name__ == "__main__":
    run_batch()
