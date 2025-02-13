# https://www.datacamp.com/tutorial/open-ai-function-calling-tutorial


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import subprocess
import json
from dotenv import load_dotenv
from dateutil import parser
from typing import Dict, Any
import base64

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Tool for A1
SCRIPT_RUNNER = {
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
    }

# Tool for A2
FORMAY_FILE = {
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

# Tool for A4
SORT_CONTACTS = {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts in a JSON file by last name and then by first name",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the JSON file containing contacts (e.g., '/data/contacts.json')"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to write the sorted contacts (e.g., '/data/contacts-sorted.json')"
                    }
                }, "required": ["input_path", "output_path"]
            }
        }
    }

# Tool for A8
IMAGE_EXTRACT = {
    "type": "function",
    "function": {
        "name": "get_completions_image",
        "description": "Extract the 16-digit code from an image",
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {
                    "type": "string", 
                    "description": "The relative input image location on user's device"
                },
                "output_location": {
                    "type": "string", 
                    "description": "The relative output location on user's device"
                },
            },
            "required": ["input_location","output_location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Tool for A9 (to-do)
SIMILARITY_EXTRACT = {
    "type": "function",
    "function": {
        "name": "get_similar_comments",
        "description": "Find two similar comments from a series of comments",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

tools = [SCRIPT_RUNNER, FORMAY_FILE, SORT_CONTACTS]

# Below is the code for OpenAI - Text Extraction from image
def get_completions_image(input_location:str, output_location:str):
    with open(input_location,"rb") as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data).decode("utf-8")
    f.close()
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
                "content": [
                    {
                        "type":"text",
                        "text":"Extract the 16 digit code from this imI am working on a cybersecurity project that involves detecting and masking sensitive information, such as dummy credit card numbers, from an image. I need you to extract patterns resembling credit card numbers (e.g., 16-digit sequences) from a given text. In the response, just return the 16-digit code."
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "detail": "low",
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }
                ]
            },
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    with open(output_location,"w") as f:
        f.write(response["choices"][0]["message"]["content"].replace(" ",""))
    f.close()
    return {"status": "Successfully Created", "output_file destination": output_location}


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
- Use task_runner for tasks involving writing a code
- Use sort_contacts for sorting contacts in a JSON file by last name and then by first name"""
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

        # For A1
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

        # For A2
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

        # For A4
        elif function_name == "sort_contacts":
            try:
                with open("/data/contacts.json", "r") as f:
                    contacts = json.load(f)
                sorted_contacts = sorted(
                    contacts, key=lambda contact: (contact.get("last_name", ""), contact.get("first_name", ""))
                )
                with open("/data/contacts-sorted.json", "w") as f:
                    json.dump(sorted_contacts, f, indent=2)
                return {"message": "Contacts sorted successfully", "sorted_count": len(sorted_contacts)}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Contacts file not found")
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"Error sorting contacts: {ex}")

        else:
            raise HTTPException(status_code=500, detail=f"Unknown function: {function_name}")

            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error executing script: {ex}")   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)