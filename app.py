# https://www.datacamp.com/tutorial/open-ai-function-calling-tutorial


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
        "name": "function",
        "function":{
            "name": "script_runner",
            "description": "Install a package and run a script from an URL with provided arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the script to run"
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Arguments to pass to the script"
                    }   

                }
            }
        }
    }
]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/read")
def read_file(path:str):
    return {"path": path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)