import os
import subprocess
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import sqlite3
import requests
from openai import OpenAI

app = FastAPI()

# Configuration
DATA_DIR = "/data"
ALLOWED_PATHS = [os.path.join(DATA_DIR, ""), "/data"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_allowed_path(path: str) -> bool:
    resolved_path = os.path.abspath(path)
    return any(resolved_path.startswith(allowed) for allowed in ALLOWED_PATHS)

def parse_task_with_llm(task_description: str) -> List[Dict[str, Any]]:
    prompt = f"""Parse this task into executable steps: {task_description}
    Output JSON steps with 'type' (command, file_op, llm_extract, llm_vision, embedding, sql) and 'parameters'."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["steps"]

def execute_steps(steps: List[Dict[str, Any]]):
    for step in steps:
        try:
            if step["type"] == "command":
                subprocess.run(step["parameters"]["command"], shell=True, check=True)
            elif step["type"] == "file_op":
                # Implement file operations
                pass
            # Add handlers for other step types
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
async def run_task(task: str = Query(...)):
    try:
        steps = parse_task_with_llm(task)
        execute_steps(steps)
        return {"status": "success"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str = Query(...)):
    if not is_allowed_path(path):
        raise HTTPException(status_code=403, detail="Path not allowed")
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example handlers for specific tasks (simplified)
def count_weekdays(file_path: str, weekday: str, output_path: str):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    target = weekdays.index(weekday)
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            date_str = line.strip()
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date.weekday() == target:
                count += 1
    with open(output_path, "w") as f:
        f.write(str(count))

# Similar handler functions for other tasks would be implemented here