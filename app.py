# https://www.datacamp.com/tutorial/open-ai-function-calling-tutorial


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import subprocess
import json
from dotenv import load_dotenv
load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tools = [
    {
        "type": "function",
        "function":{
            "name": "script_runner",
            "description": "Install a package and run a script from an URL with provided arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the script to run"
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of arguments to pass to the script"
                    },   

                }, "required": ["script_url", "args"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a file using Prettier with specified version",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to format"
                    },
                "prettier_version": {
                    "type": "string",
                    "description": "Prettier version to use (e.g. '3.4.2')"
                    }
                }, "required": ["path", "prettier_version"]
            }
        }
    }
]

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/read")
def read_file(path:str):
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
@app.post("/run")
def task_runner(task:str):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
 
    data = {   
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": task
            },
            {
                "role": "system",
                "content": """You are an assistant who has to do a variety of tasks.
- Use script_runner for installing packages and running scripts from URLs
- Use format_file for formatting files with Prettier
- Use task_runner for tasks involving writing a code"""
            }
        ],
        "tools": tools,
        "tool_choice": "auto"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    # if not response.ok:
    #     raise HTTPException(status_code=response.status_code, detail=response.text)
    
    try:
        res_json = response.json()
        message = res_json['choices'][0]['message']
        tool_calls = message.get('tool_calls', [])
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Error parsing response: {e}\nResponse: {res_json}")
    
    try:
        function_name = message['tool_calls'][0]['function']['name'] # Extract the function name
        if function_name == "script_runner":
            arguments = message['tool_calls'][0]['function']['arguments']
            # If arguments is a string, parse it; otherwise, assume it's already a dict.
            if isinstance(arguments, str):
                func_args = json.loads(arguments)
            else:
                func_args = arguments
            script_url = func_args['script_url']
            email = func_args['args'][0]
            command = ["uv","run",script_url, email]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return {"output": result.stdout, "error": result.stderr}
        elif function_name == "format_file":
            arguments = message['tool_calls'][0]['function']['arguments']
            path = arguments['path']
            version = arguments['prettier_version']
            # Checking if the file exists
            if not os.path.exists(path):
                    raise HTTPException(status_code=400, detail=f"File not found: {path}")
            command = ["npx", f"prettier@{version}", "--write", path]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return {"output": result.stdout, "error": result.stderr}

        else:
            raise HTTPException(status_code=500, detail=f"Unknown function: {function_name}")

            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error executing script: {ex}")   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)