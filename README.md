## Explicit Planning under Compute Constraints: A Gridworld Search-Based Autonomous Agent
personal project exploring search based agents

Implemented a model-based autonomous agent using A* and BFS, and MCTS to study how planning depth and heuristic admissibility affect success, optimality, and runtime in long-horizon tasks. Built a reproducible evaluation harness with instrumentation and failure-mode analysis.

I implemented a deterministic, search-based planning agent for long-horizon tasks in a 4-neighbor, unit-cost Gridworld environment. 

## Features
- 4-neighbor, unit-cost Gridworld (no diagonals)
- BFS (oracle) and A* (with Manhattan and weighted heuristics)
- Structured evaluation and ablation harness with parameter sweeps (grid size up to 30x30, obstacle density up to 0.3)
- Failure mode analysis (no path, timeout, budget exceeded)
- Comparison tables of average/median metrics for all algorithms and settings
- Clean, reproducible metrics and plots
- Minimal, principled Python code (no deep ML stack)
- MCTS (Monte Carlo Tree Search) is implemented and evaluated for grid sizes up to 30x30 and density 0.3 as an additional/experimental algorithm; not the primary focus.


## Usage
- Define a planning task as a JSON dictionary (see `instructions.txt` for schema)
- Use the runner to execute tasks and log results
- Analyze results and generate plots for ablations
  - Results and plots are saved in `results/exp_2025-12-29_01/`
  - `nodes_expanded_vs_optimality_gap.png` and `runtime_vs_grid_size.png` for main plots
  - `failure_modes.csv` for failure mode analysis (no path, timeout, budget exceeded)
  - `comparison_table.csv` for average/median metrics by algorithm and setting

## Research Note
I study explicit planning under compute constraints in deterministic grid environments. Using BFS as an oracle, I analyze how heuristic admissibility and search aggressiveness affect solution quality and runtime. Eesults show that weighted heuristics significantly reduce search cost at the expense of optimality, with failure modes emerging at higher obstacle densities. MCTS is included as an experimental baseline and evaluated for grid sizes up to 30x30 and density 0.3. All results are reproducible and summarized in the results directory.
  - Sweeps are configurable; results reported in this repo cover up to 30x30 grids and density 0.3.

## Requirements (Core)
- Python 3.8+
- numpy
- matplotlib
- pandas (optional)
- pytest (optional)

## Optional: LLM/Agentic Integration
LLM integration is not used in core experiments and is provided as an optional demo only.
- To enable LLM/agentic features, install dependencies from `requirements-llm.txt`:
  - torch
  - transformers
  - langchain
  - langchain-community
  - huggingface_hub

**LLM/Agentic Demo (Optional):**
  - Run `python3 llm_agentic_demo.py` to see a local LLM (e.g., Flan-T5-large) reason about a planning task and trigger the planner tool.
  - Output is saved to `llm_planner_output.txt`.
  - You can swap in any Hugging Face instruction-tuned model (see script for details).
  - LLM wrapper is optional and not used in core experiments.



## Next Steps
- Try larger or more advanced instruction-tuned models (e.g., Mistral-7B-Instruct, Llama-2-7b-chat, Zephyr-7b-beta) if you have the hardware (optional).
- Experiment with more complex agentic pipelines or real function-calling LLMs as they become available (optional).
- Extend the planner or environment for new research directions.
## Results
 - All results, plots, and tables are in `results/exp_2025-12-29_01/` after running the batch and analysis scripts.
 - See `failure_modes.csv` and `comparison_table.csv` for analysis.
