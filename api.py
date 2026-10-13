# api.py —— FastAPI 服务入口
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import init as rag_init
from main import run_agent_with_history, SYSTEM_PROMPT

app = FastAPI(title="RAG + Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    rag_init()


class Query(BaseModel):
    question: str
    history: Optional[list] = Field(default_factory=list)


@app.post("/chat")
def chat(q: Query):
    try:
        history = [{"role": "system", "content": SYSTEM_PROMPT}] + q.history
        history.append({"role": "user", "content": q.question})
        ans = run_agent_with_history(history, verbose=False)
        return {"answer": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}