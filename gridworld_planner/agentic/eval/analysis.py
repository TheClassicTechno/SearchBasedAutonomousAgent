import numpy as np
import pandas as pd
"""
Analysis and plotting for Gridworld planning experiments.
"""
import json
import os
import matplotlib.pyplot as plt

def load_results(results_path):
    # Load experiment results from a JSONL file, one result per line
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

def summarize_failures(results, out_path=None):
    # Summarize failure modes by algorithm, heuristic, grid size, and density
    failure_modes = {}
    for r in results:
        if not r["success"]:
            key = (r["algorithm"], r["heuristic"], r["grid_width"], r["obstacle_density"])
            reason = r["termination_reason"]
            failure_modes.setdefault(key, {}).setdefault(reason, 0)
            failure_modes[key][reason] += 1
    # Format the summary as CSV lines
    lines = []
    for key, reasons in failure_modes.items():
        algo, heuristic, width, density = key
        for reason, count in reasons.items():
            lines.append(f"{algo},{heuristic},{width},{density},{reason},{count}")
    header = "algorithm,heuristic,grid_width,obstacle_density,reason,count"
    report = "\n".join([header] + lines)
    
    if out_path:
        with open(out_path, "w") as f:
            f.write(report)
    else:
        print(report)
    return report

def comparison_table(results, out_path=None):
    # Create a summary table comparing algorithms and heuristics
    df = pd.DataFrame(results)
    # Only consider successful runs
    df = df[df["success"]]
    group_cols = ["algorithm", "heuristic", "grid_width", "obstacle_density"]
    metrics = ["path_cost", "optimality_gap", "nodes_expanded", "runtime_ms"]
    summary = df.groupby(group_cols)[metrics].agg(["mean", "median"]).reset_index()
    # Output to CSV or print
    if out_path:
        summary.to_csv(out_path, index=False)
    else:
        print(summary)
    return summary

def plot_nodes_expanded_vs_optimality_gap(results, out_path=None):
    # Prepare data for plotting nodes expanded vs. optimality gap
    xs = []
    ys = []
    algos = []
    for r in results:
        if r["success"] and r["optimality_gap"] is not None:
            xs.append(r["nodes_expanded"])
            ys.append(r["optimality_gap"])
            algos.append(f"{r['algorithm']}_{r['heuristic']}")
    plt.figure(figsize=(8,6))
    for algo in set(algos):
        idx = [i for i, a in enumerate(algos) if a == algo]
        plt.scatter([xs[i] for i in idx], [ys[i] for i in idx], label=algo, alpha=0.7)
    plt.xlabel("Nodes Expanded")
    plt.ylabel("Optimality Gap")
    plt.title("Nodes Expanded vs Optimality Gap")
    plt.legend()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

def plot_runtime_vs_grid_size(results, out_path=None):
    xs = []
    ys = []
    algos = []
    for r in results:
        if r["success"]:
            xs.append(r["grid_width"])
            ys.append(r["runtime_ms"])
            algos.append(f"{r['algorithm']}_{r['heuristic']}")
    plt.figure(figsize=(8,6))
    for algo in set(algos):
        idx = [i for i, a in enumerate(algos) if a == algo]
        plt.plot([xs[i] for i in idx], [ys[i] for i in idx], marker='o', label=algo)
    plt.xlabel("Grid Size (width)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Grid Size")
    plt.legend()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "exp_2025-12-29_01", "runs.jsonl")
    out_dir = os.path.dirname(results_path)
    results = load_results(results_path)
    plot_nodes_expanded_vs_optimality_gap(results, out_path=os.path.join(out_dir, "nodes_expanded_vs_optimality_gap.png"))
    plot_runtime_vs_grid_size(results, out_path=os.path.join(out_dir, "runtime_vs_grid_size.png"))
    summarize_failures(results, out_path=os.path.join(out_dir, "failure_modes.csv"))
    comparison_table(results, out_path=os.path.join(out_dir, "comparison_table.csv"))
    print(f"Plots and tables saved to {out_dir}")
