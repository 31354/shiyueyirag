# api.py —— FastAPI 多 Agent 服务入口
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 引入 RAG 初始化 和 多 Agent 状态图
from rag import init as rag_init
from multi_agent import app as agent_app, AgentState

app = FastAPI(title="RAG + Multi-Agent API", version="1.0.0")

# 跨域配置，方便前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # 启动时只初始化一次 RAG 向量库，避免每次请求都卡顿
    rag_init()


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(q: Query):
    try:
        # 初始化多 Agent 的全局状态
        initial_state = {
            "task": q.question,
            "retrieved_docs": "",
            "calculation": "",
            "draft_answer": "",
            "review_passed": False,
            "revision_count": 0
        }

        # 调用 LangGraph 状态机执行完整流程
        final_state = agent_app.invoke(initial_state)

        # 提取 Writer 最终生成的回答
        return {
            "answer": final_state.get("draft_answer", "未生成有效回答"),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多 Agent 运行失败：{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "agent_type": "multi-agent"}