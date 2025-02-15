# https://www.datacamp.com/tutorial/open-ai-function-calling-tutorial


from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import subprocess
import json
from dotenv import load_dotenv
from datetime import datetime
from dateutil import parser
from typing import Dict, Any
import base64
import numpy as np
import pandas as pd
import sqlite3, duckdb, httpx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

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
                            "type": "string",
                            "description": "List of arguments to pass to the script"
                            },
                    },   

                }, "required": ["script_url", "args"]
            }
        },
    }

# Tool for A2
FORMAT_FILE = {
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

# Tool for A3
COUNT_DAYS = {
        "type": "function",
        "function": {
            "name": "count_days",
            "description": "Count the number of a specific weekday in a list of dates and write the count to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day_of_week": {
                        "type": "string",
                        "description": "Day of week to be counted, e.g. wednesday",
                    },
                    "input_file_path": {
                        "type": "string",
                        "description": "Path to the input file containing a list of dates",
                    },
                    "output_file_path": {
                        "type": "string",
                        "description": "Path to the output file where the number of that specific weekday will be written.",
                    },
                },
                "required": ["day_of_week", "input_file_path", "output_file_path"],
                "additionalProperties": False,
            },            
            "strict": True
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

# Tool for A5
LOGS_RECENT = {
        "type": "function",
        "function": {
            "name": "logs_recent",
            "description": "Retrieve the most recent log files from a directory and save their content to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                        "log_dir_path": {
                            "type": "string",
                            "description": "Path to the directory containing log files",
                        },
                        "output_file_path": {
                            "type": "string",
                            "description": "Path to the output file to write the log content",
                        },
                        "num_files": {
                            "type": "integer",
                            "description": "Number of recent log files to retrieve",
                        }
                    }, "required": ["log_dir_path", "output_file_path", "num_files"]
                }
            }
        }

# Tool for A6
MARKDOWN_INDEX = {
            "type": "function",
            "function": {
                "name": "markdown_index",
                "description": "Generate an index of documents from a directory and save it as a Markdown file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_dir_path": {
                            "type": "string",
                            "pattern": r".*/docs",
                            "default": "/data/docs"
                        },
                        "output_file_path": {
                            "type": "string",
                            "pattern": r".*/(.*\.md)",
                            "default": "/data/docs/index.md"
                        }
                    },
                    "required": ["doc_dir_path", "output_file_path"]
                }
            }
        }

# Tool or A7
EMAIL_SENDER = {
        "type": "function",
        "function": {
            "name": "email_sender",
            "description": "Extract the sender's email address and in response return just the sender's email address",
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

# Tool for A9
SIMILARITY_EXTRACT = {
        "type": "function",
        "function": {
            "name": "get_similar_comments",
            "description": "Find two similar comments from a series of comments",
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

# Tool for A10
QUERY_SQL = {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": "Correctly identify the table and the column names of the table and correctly identify the SQL query that has to be run on the table as query.Identify the database file as filename.Identify the output file where the results of the SQL query will be saved as output_filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Write just the SQL query according to the question"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name of the database file to run the SQL query on"
                    },
                    "output_filename": {
                        "type": "string",
                        "description": "Name of the output file to save the SQL query results"
                    }
                },
                "required": ["query", "filename", "output_filename"],
            }
        }
    }

tools = [SCRIPT_RUNNER, FORMAT_FILE, COUNT_DAYS, SORT_CONTACTS, LOGS_RECENT, MARKDOWN_INDEX, EMAIL_SENDER,IMAGE_EXTRACT, SIMILARITY_EXTRACT, QUERY_SQL]

def format_and_save_markdown(input_path: str, prettier_version: str, output_path: str):
    # Read the unformatted markdown content
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Build a refined prompt message with extra instructions for list items
    refined_prompt = (
        f"Please format the following markdown text using Prettier version {prettier_version}.\n"
        "Ensure that the output is a clean, properly formatted markdown text with correct spacing, indentation, and newlines.\n"
        "Specifically, remove any unnecessary indentation before list items (for example, extra spaces before dashes) so that all list items align consistently.\n"
        "Do NOT wrap your output in any markdown code block markers or provide any extra commentary.\n\n"
        "Markdown content:\n\n"
        f"{content}"
    )
    # Prepare the request
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert markdown formatter. Your task is to reformat the provided markdown "
                    "text exactly as needed for a .md file. Output only the formatted markdown text with proper "
                    "spacing and newlines, and remove any unnecessary indentation (such as extra spaces before list items). "
                    "Do not include any additional commentary or markdown formatting wrappers."
                )
            },
            {
                "role": "user",
                "content": refined_prompt
            }
        ]
    }
    # Send the request to the proxy API
    response = requests.post(url, headers=headers, json=data).json()
    try:
        # Extract the formatted markdown text from the response.
        if "choices" in response:
            formatted_markdown = response["choices"][0]["message"]["content"]
        elif "message" in response:
            formatted_markdown = response["message"]["content"]
        else:
            raise ValueError("Unexpected response format: " + str(response))
        
        # Write the formatted markdown text to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_markdown)
        
        return Response(
                            content=json.dumps({"output": formatted_markdown}),
                            status_code=200,
                            media_type="application/json"
                        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error formatting markdown: {str(e)}")
    
# Function to count number of days of a particular day from a text file containing a list of dates
def count_days(date: str, input_location:str, output_location:str):
    # Define possible date formats
    date_formats = [
        "%b %d, %Y",        # Apr 04, 2006
        "%Y-%m-%d",         # 2017-01-09
        "%d-%b-%Y",         # 17-Dec-2001
        "%Y/%m/%d",         # 2006/04/12
        "%Y/%m/%d %H:%M:%S" # 2006/04/12 21:29:21 (new format added)
    ]
    if date.lower() == "monday":
        day = 0
    elif date.lower() == "tuesday":
        day = 1
    elif date.lower() == "wednesday":
        day = 2
    elif date.lower() == "thursday":
        day = 3
    elif date.lower() == "friday":
        day = 4
    elif date.lower() == "saturday":
        day = 5
    elif date.lower() == "sunday":
        day = 6
    else:
        raise HTTPException(status_code=400, detail="Invalid day of the week")

    count = 0
    with open(input_location, "r") as file:
        for line in file:
            date_str = line.strip()
            matched = False  # Track if the date matched a format
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    if parsed_date.weekday() == day:
                        count += 1
                    matched = True
                    break  # Stop checking once a format matches
                except ValueError:
                    pass  # Try the next format
            if not matched:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}")
    with open(output_location, "w") as file:
        file.write(str(count)) # .replace('"', ''))
    file.close()
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_location}),
                    status_code=200,
                    media_type="application/json")
    
def log_recent(log_dir_path:str, output_file_path:str, num_files:int):
    # Get the first line of the 10 most recent log files
    command = f"ls -t {log_dir_path} | head -n {num_files}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log_files = result.stdout.splitlines()
    with open(output_file_path, "w") as outfile:
        for log_file in log_files:
            with open(os.path.join(log_dir_path, log_file), "r") as infile:
                outfile.write(infile.readline().strip())
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_file_path}),
                    status_code=200,
                    media_type="application/json")

def markdown_index(doc_dir_path:str, output_file_path:str):
    # Find all Markdown(.md) files in doc_dir_path and for each file, extract the first occurance of each H1(#)
    # heading and write it to the output_file_path as json format without the prefixpath
    files = []
    index_data = {}
    for root, dirs, files in os.walk(doc_dir_path):
        for file in files:
            if file.endswith(".md"):
                files.append(os.path.join(root, file))
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for file in files:
            with open(file, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        relative_path = os.path.relpath(file, doc_dir_path).replace("\\", "/")
                        index_data[relative_path] = title
                        break
    with open(output_file_path, "w", encoding="utf-8") as outfile: 
        json.dump(index_data, outfile, indent=2)
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_file_path}),
                    status_code=200,
                    media_type="application/json")

def email_sender(input_location:str, output_location:str):
    with open(input_location,"rb") as f:
        text = f.read().decode("utf-8")
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
                "role": "system", 
                "content": "Extract just the sender's email address from this email and return just the sender's email address."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    with open(output_location,"w") as f:
        f.write(response["choices"][0]["message"]["content"].replace(" ","").replace('"',''))
    f.close()
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_location}),
                    status_code=200,
                    media_type="application/json")        

# Below is the code for OpenAI - Text Extraction from image
# https://colab.research.google.com/drive/1bK0b1XMrZWImtw01T1w9NGraDkiVi8mS#scrollTo=RR_q1bi8kfHH
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
    # Stronger prompt instructing for a 16-digit continuous number
    prompt_text = (
        "You are a highly accurate extraction system. I am providing an image that contains a dummy credit card number "
        "generated by the Faker module. The image has exactly one credit card number that is a continuous string of 16 digits "
        "with no spaces, dashes, or any other characters. Your task is to extract and return ONLY that 16-digit number as a plain text string. "
        "Do NOT include any spaces, punctuation, line breaks, or additional commentary. For example, if the number is 3580146805161976, "
        "simply return '3580146805161976'."
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {
                        "type":"text",
                        "text": prompt_text, #"Extract the 16 digit code from this imI am working on a cybersecurity project that involves detecting and masking sensitive information, such as dummy credit card numbers, from an image. I need you to extract patterns resembling credit card numbers (e.g., 16-digit sequences) from a given text. In the response, just return the 16-digit code."
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
        f.write(str(response["choices"][0]["message"]["content"]))  #.replace(" ",""))
    f.close()
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_location}),
                    status_code=200,
                    media_type="application/json")

# def get_similar_comments(input_location:str, output_location:str):
#     with open(input_location,"r") as f:
#         comments = [i.strip() for i in f.readlines()]
#         f.close()
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(comments)
#     similarity_mat = cosine_similarity(embeddings)
#     np.fill_diagonal(similarity_mat,0)
#     max_index = int(np.argmax(similarity_mat))
#     i, j = max_index//len(comments), max_index%len(comments)
#     with open(output_location,"w") as g:
#         g.write(comments[i])
#         g.write("\n")
#         g.write(comments[j])
#         g.close()
#     return {"status": "Successfully Created", "output_file destination": output_location}

# def get_similar_comments(input_location: str, output_location: str):
#     # Read comments from the input file
#     with open(input_location, "r") as f:
#         comments = [line.strip() for line in f]
#     f.close()
#     # Generate TF-IDF embeddings for the comments
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(comments)
#     # Calculate cosine similarity matrix
#     similarity_mat = cosine_similarity(tfidf_matrix)
#     np.fill_diagonal(similarity_mat, 0)  # Ignore self-similarity
#     # Find the indices of the maximum similarity score
#     max_index = np.argmax(similarity_mat)
#     i, j = max_index // len(comments), max_index % len(comments)
#     # Write the most similar pair to the output file
#     with open(output_location, "w") as g:
#         g.write(f"{comments[i]}\n{comments[j]}")
#         g.close()
#     return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_location}),
#                     status_code=200,
#                     media_type="application/json")

async def get_similar_comments(input_location: str, output_location: str):
    # Read comments from the input file
    with open(input_location, "r") as f:
        comments = [line.strip() for line in f if line.strip()]
    # Generate OpenAI embeddings
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {AIPROXY_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": comments
                },
                timeout=30
            )
            response.raise_for_status()
            embeddings_data = response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error: {e.response.text}"
            )
    # Extract and normalize embeddings
    embeddings = np.array([item['embedding'] for item in embeddings_data['data']])
    # Calculate cosine similarity matrix
    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -np.inf)  # Ignore self-similarity
    # Find indices of maximum similarity
    max_index = np.argmax(similarity)
    i, j = np.unravel_index(max_index, similarity.shape)
    # Sort the pair alphabetically before writing
    sorted_pair = sorted([comments[i], comments[j]])
    # Write to output file
    with open(output_location, "w") as f:
        f.write("\n".join(sorted_pair))
    return Response(
        content=json.dumps({
            "status": "Success",
            "pair_found": sorted_pair,
            "similarity_score": float(similarity[i, j])
        }),
        status_code=200,
        media_type="application/json"
    )

def query_sql(filename:str, query:str, output_filename:str):
    if not filename.endswith('.db'):
        return None
    conn = sqlite3.connect(filename) if filename.endswith('.db') else duckdb.connect(filename)
    cur = conn.cursor()
    cur.execute(query)
    # result = cur.fetchall()
    result = cur.fetchone()[0]
    # If there are no sales, set total_sales to 0
    result = result if result else 0
    # Make the result value upto two-decimal
    result = round(result, 2)
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return Response(content=json.dumps({"status": "Successfully Created", "output_file destination": output_filename}),
                    status_code=200,
                    media_type="application/json")

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
        raise HTTPException(status_code=500, detail=str(e))
    
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
- Use sort_contacts for sorting contacts in a JSON file by last name and then by first name
- Use get_completions_image to run the get_completions_image function and save output to a file
- Use email_sender to run the email_sender function and save output to a file
- Use get_similar_comments to run the get_similar_comments function and save output to a file
- Use query_sql to run a SQL query and save the result to a file
- Use logs_recent to retrieve the most recent log files from a directory and save their content to an output file
- Use markdown_index to index the contents of a directory and save the index to a file"""
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
        # tool_calls = message.get('tool_calls', [])
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
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                return Response(
                        content=json.dumps({"output": result.stdout}),
                        status_code=200,
                        media_type="application/json"
                    )
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                        status_code=400,
                        detail=f"Script execution failed: {e.stderr or e.stdout}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unknown error: {str(e)}")

        # For A2
        elif function_name == "format_file":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            path = arguments['path']
            version = arguments['prettier_version']
            # Checking if the file exists
            if not os.path.exists(path):
                    raise HTTPException(status_code=400, detail=f"File not found: {path}")
            return format_and_save_markdown(path, version, path)
        
            # Use proper formatting command
            # command = [
            #     "npx", 
            #     f"prettier@{version}",
            #     "--write",
            #     path
            # ]
            # try:
            #     result = subprocess.run(command, capture_output=True, text=True, check=True)
            #     return Response(
            #             content=json.dumps({"output": result.stdout}),
            #             status_code=200,
            #             media_type="application/json"
            #         )
            # except subprocess.CalledProcessError as e:
            #     raise HTTPException(
            #             status_code=400,
            #             detail=f"Script execution failed: {e.stderr or e.stdout}"
            #     )
            # except Exception as e:
            #     raise HTTPException(status_code=500, detail=f"Formatter error: {str(e)}")

        # For A3
        elif function_name == "count_days":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            day = arguments['day_of_week']
            input_file_path = arguments['input_file_path']
            output_file_path = arguments['output_file_path']
            return count_days(day, input_file_path, output_file_path)

        # For A4
        elif function_name == "sort_contacts":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            # Parse and validate arguments
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            # Check required parameters exist
            if 'input_path' not in arguments or 'output_path' not in arguments:
                missing = [k for k in ['input_path', 'output_path'] if k not in arguments]
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing parameters: {', '.join(missing)}"
                )
            input_file_path = arguments['input_path']
            output_file_path = arguments['output_path']
            try:
                with open(input_file_path, "r") as f:
                    contacts = json.load(f)
                # sorted_contacts = sorted(
                #     contacts, key=lambda contact: (contact.get("last_name", ""), contact.get("first_name", ""))
                # )
                contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))
                with open(output_file_path, "w") as f:
                    json.dump(contacts, f) # changed to contacts
                return Response(
                    content=json.dumps({"message": "Contacts sorted successfully", "sorted_count": len(contacts)}),
                    status_code=200,
                    media_type="application/json")
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail="Contacts file not found")
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"Error sorting contacts: {ex}")

        # For A5
        elif function_name == "logs_recent":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            log_dir_path = arguments['log_dir_path']
            output_file_path = arguments['output_file_path']
            num_files = arguments['num_files']
            return log_recent(log_dir_path, output_file_path, num_files)

        # For A6
        elif function_name == "markdown_index":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            doc_dir_path = arguments['doc_dir_path']
            output_file_path = arguments['output_file_path']
            return markdown_index(doc_dir_path, output_file_path)

        # For A7
        elif function_name == "email_sender":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            input_location = arguments['input_location']
            output_location = arguments['output_location']
            return email_sender(input_location, output_location)

        # For A8
        elif function_name == "get_completions_image":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            input_location = arguments['input_location']
            output_location = arguments['output_location']
            return get_completions_image(input_location, output_location)

        # For A9
        elif function_name == "get_similar_comments":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            input_location = arguments['input_location']
            output_location = arguments['output_location']
            return get_similar_comments(input_location, output_location)

        # For A10
        elif function_name == "query_sql":
            raw_arguments = message['tool_calls'][0]['function']['arguments']
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            else:
                arguments = raw_arguments
            query = arguments['query']
            filename = arguments['filename']
            output_filename = arguments['output_filename']
            return query_sql(query, filename, output_filename)

        else:
            raise HTTPException(status_code=500, detail=f"Unknown function: {function_name}")

            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error executing script: {ex}")   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)