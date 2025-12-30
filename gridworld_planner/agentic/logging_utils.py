"""
Minimal JSONL logger for planning runs and traces.
"""
import json
import os
from typing import Any, Dict

class JsonlLogger:
    def __init__(self, out_dir: str):
        # Initialize the logger and ensure output directory exists
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.runs_path = os.path.join(out_dir, "runs.jsonl")

    def log_run(self, result: Dict[str, Any]) -> None:
        # Append a single run result to the runs.jsonl file
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def log_trace(self, task_id: str, trace: Dict[str, Any]) -> None:
        # Save a trace for a specific task as a separate JSON file
        traces_dir = os.path.join(self.out_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)
        path = os.path.join(traces_dir, f"{task_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
