"""
Demo: Use an LLM (via LangChain) to call the Gridworld planner as a tool.
 shows how an LLM can invoke your search agent to solve a planning task.
"""
from langchain_core.tools import Tool
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.agents import initialize_agent, AgentType
from agentic.tools import solve_gridworld_task
import os



# Wrap your planner tool for LangChain agent
planner_tool = Tool(
    name="solve_gridworld_task",
    func=solve_gridworld_task,
    description="Solve a Gridworld planning task using A*, BFS, or MCTS. Input: task_json dict. Output: RunResult dict."
)

# Load a local Hugging Face model and tokenizer (change model name as needed)
model_name = "google/flan-t5-large"  # Instruction-tuned
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text-generation pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

# Wrap the pipeline for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Create an agent with your tool
agent = initialize_agent(
    tools=[planner_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Example user prompt for the agent
task_dict = {
    "task_id": "demo_llm_001",
    "seed": 123,
    "grid": {
        "width": 5,
        "height": 5,
        "obstacle_density": 0.1,
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

import re

def custom_agent_loop(llm, planner_tool, task_dict):
    """
    Simulate tool-calling: LLM is prompted to call the planner tool by outputting a special command.
    """
    prompt = (
        "You are an AI assistant. If you see a command like CALL_PLANNER(task_json), "
        "you must call the Python planner tool with the given task_json and return the result.\n"
        f"Here is a planning task: {task_dict}\n"
        "If you want to solve it, output: CALL_PLANNER({task_dict})\n"
        "Otherwise, explain what you would do."
    )
    llm_response = llm(prompt)
    print("LLM raw response:\n", llm_response)
    # Look for the special command
    match = re.search(r"CALL_PLANNER\((.*?)\)", llm_response, re.DOTALL)
    if match:
        # In a real system, you'd parse JSON safely. Here, we use the known dict.
        print("\n[LLM requested planner tool call]")
        result = planner_tool({"task_json": task_dict})
        print("Planner tool result:", result)
        with open("llm_planner_output.txt", "w") as f:
            f.write("LLM Response:\n" + llm_response + "\n\nPlanner Tool Result:\n" + str(result))
    else:
        print("\n[LLM did not request planner tool call]")
        with open("llm_planner_output.txt", "w") as f:
            f.write("LLM Response:\n" + llm_response)

if __name__ == "__main__":
    # Use the custom agent loop for simulated tool-calling
    custom_agent_loop(llm, solve_gridworld_task, task_dict)

