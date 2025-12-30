"""
LangChain tool wrapper for Gridworld planning agent.
Wraps the core runner as a LangChain tool for evaluation harness.
"""
from langchain_core.tools import tool
from agentic.eval.runner import run_task_from_dict

@tool
def solve_gridworld_task(task_json: dict) -> dict:
    """
    Solve a Gridworld planning task using the specified algorithm/heuristic.
    Input: TaskSpec dict
    Output: RunResult dict
    """
    # Call the core runner to solve the task and return the result
    return run_task_from_dict(task_json)
