from flask import Flask, request, jsonify
import openai
import sqlite3
import subprocess

app = Flask(__name__)

# Configure OpenAI API key
openai.api_key = "YOUR_API_KEY"

def execute_task(task_description):
    # Use LLM to classify the task into A1-A10 based on examples
    prompt = f"Classify the following task description into one of the predefined tasks (A1 to A10):\n{task_description}"
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}])
    task_id = response.choices[0].message['content'].strip()

    # Execute the corresponding task
    if task_id == 'A1':
        return handle_task_a1()
    elif task_id == 'A2':
        return handle_task_a2()
    # ... similar for A3 to A10
    else:
        return f"Task {task_id} not recognized", 400

def handle_task_a1(email):
    try:
        # Check if uv is installed
        subprocess.check_output(['uv', '--version'])
        # Run the script with email argument
        result = subprocess.run(['python3', 'script.py', email], capture_output=True, text=True)
        return result.stdout, 200
    except Exception as e:
        return str(e), 500

def handle_task_a2():
    try:
        # Format file using black
        subprocess.check_call(['black', 'file.py'])
        return "File formatted successfully", 200
    except Exception as e:
        return str(e), 500

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task')
    result, status_code = execute_task(task)
    return jsonify({"result": result}), status_code

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content, 200
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)