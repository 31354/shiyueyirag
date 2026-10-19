# api.py —— FastAPI 服务入口（多 Agent 升级版）
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import init as rag_init
# 修改 1：引入你的多 Agent 状态图
from multi_agent import app as agent_app, AgentState

app = FastAPI(title="RAG + Multi-Agent API")

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


@app.post("/chat")
def chat(q: Query):
    try:
        # 修改 2：初始化多 Agent 的全局状态
        initial_state = {
            "task": q.question,
            "retrieved_docs": "",
            "calculation": "",
            "draft_answer": "",
            "review_passed": False,
            "revision_count": 0
        }

        # 修改 3：调用 LangGraph 工作流执行任务
        final_state = agent_app.invoke(initial_state)

        # 修改 4：从最终状态中提取 Writer 生成的草稿返回
        return {
            "answer": final_state["draft_answer"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}