import subprocess
from fastapi import FastAPI, HTTPException, Query
import os

app = FastAPI()

# Directory to store files
UPLOAD_DIR = "/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/run")
async def run_task(task: str = Query(...)):
    try:
        # Parse the task and check if it matches formatting a markdown file with prettier
        if "Format" in task and "prettier" in task.lower():
            # Extract the path from the task string (e.g., "/data/format.md")
            parts = task.split("with prettier")
            file_path = parts[0].replace("Format", "").strip()
            
            # Run prettier command on the file
            result = subprocess.run(
                ["prettier", "--write", file_path], 
                capture_output=True, text=True
            )

            if result.returncode == 0:
                return {"status": "success", "message": "File formatted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Prettier formatting failed")
        else:
            raise HTTPException(status_code=400, detail="Task not recognized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, path.lstrip("/"))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
            return {"content": content}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
