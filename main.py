from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import research_paper_assistant
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000/",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIResponse(BaseModel):
    response: str

class AIRequest(BaseModel):
    input: str

@app.post("/ask-ai", response_model=AIResponse)
def ask_ai(request: AIRequest):
    response_text = research_paper_assistant(request.input)
    response = response_text.content
    print(response)
    return {"response": response}