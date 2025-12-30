"""
Runner for executing planning tasks from a task dictionary.
"""
import time
from agentic.env.gridworld import Gridworld
from agentic.env.generators import generate_obstacles
from agentic.search.astar import astar_search
from agentic.search.bfs import bfs_search
from agentic.search.mcts import mcts_search
from agentic.search.heuristics import manhattan, weighted_manhattan
from agentic.eval.metrics import optimality_gap

def run_task_from_dict(task_json):
    # Unpack task configuration
    grid = task_json["grid"]
    planner = task_json["planner"]
    eval_cfg = task_json.get("eval", {})

    # Extract grid and planner parameters
    width = grid["width"]
    height = grid["height"]
    start = tuple(grid["start"])
    goal = tuple(grid["goal"])
    seed = task_json.get("seed", 0)
    density = grid.get("obstacle_density", 0.0)
    obstacles = grid.get("obstacles")
    # Generate obstacles if not provided
    if obstacles is None:
        obstacles = generate_obstacles(width, height, density, seed, start, goal)
    else:
        obstacles = [tuple(x) for x in obstacles]

    # Create the Gridworld environment
    env = Gridworld(width, height, obstacles, start, goal)
    algorithm = planner["algorithm"]
    heuristic_name = planner.get("heuristic", "manhattan")
    weight = planner.get("weight", 1.0)
    max_expansions = planner.get("max_expansions", 200000)
    timeout_ms = planner.get("timeout_ms")

    # Select and run the appropriate planning algorithm
    if algorithm == "astar":
        heuristic_fn = manhattan if heuristic_name == "manhattan" else weighted_manhattan
        t0 = time.time()
        path, nodes_expanded, max_frontier_size, reason = astar_search(env, heuristic_fn, weight, max_expansions, timeout_ms)
        runtime_ms = int((time.time() - t0) * 1000)
    elif algorithm == "bfs":
        t0 = time.time()
        path, nodes_expanded, max_frontier_size, reason = bfs_search(env, max_expansions)
        runtime_ms = int((time.time() - t0) * 1000)
    elif algorithm == "mcts":
        t0 = time.time()
        path, nodes_expanded, max_frontier_size, reason = mcts_search(env, max_iterations=1000, rollout_depth=40, timeout_ms=timeout_ms or 2000)
        runtime_ms = int((time.time() - t0) * 1000)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Collect results and compute metrics
    success = path is not None
    path_len = len(path) - 1 if path else None
    path_cost = path_len if path else None
    optimal_cost = None
    optimal_gap = None
    #  compute the optimal cost using BFS (oracle)
    if eval_cfg.get("compute_oracle_optimal") and algorithm != "bfs":
        opt_path, _, _, _ = bfs_search(env, max_expansions)
        if opt_path:
            optimal_cost = len(opt_path) - 1
            if path_cost is not None:
                optimal_gap = optimality_gap(path_cost, optimal_cost)

    result = {
        "task_id": task_json.get("task_id", ""),
        "status": "success" if success else reason,
        "algorithm": algorithm,
        "heuristic": heuristic_name,
        "weight": weight,
        "grid_width": width,
        "grid_height": height,
        "obstacle_density": density,
        "seed": seed,
        "success": success,
        "path_len": path_len,
        "path_cost": path_cost,
        "optimal_cost": optimal_cost,
        "optimality_gap": optimal_gap,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "runtime_ms": runtime_ms,
        "termination_reason": reason,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return result
