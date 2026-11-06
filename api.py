# api.py —— FastAPI 多 Agent 服务入口
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from rag import init as rag_init
from multi_agent import app as agent_app

app = FastAPI(title="RAG + Multi-Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        rag_init()
    except Exception as e:
        print(f"⚠️ RAG 初始化失败（{type(e).__name__}: {e}），API 将降级运行")


class Query(BaseModel):
    question: str
    thread_id: Optional[str] = "default_user"


class ResumeQuery(BaseModel):
    thread_id: str
    human_input: str


@app.post("/chat")
def chat(q: Query):
    config = {"configurable": {"thread_id": q.thread_id}}

    # 自动从 LangGraph 的 MemorySaver 里拿历史
    past_state = agent_app.get_state(config)
    chat_history = []
    if past_state and past_state.values:
        chat_history = past_state.values.get("chat_history", [])

    initial_state = {
        "task": q.question,
        "chat_history": chat_history,
        "rewritten_query": "",
        "retrieved_docs": "",
        "calculation": "",
        "draft_answer": "",
        "review_passed": False,
        "revision_count": 0
    }

    try:
        final_state = agent_app.invoke(initial_state, config=config)
        current_state = agent_app.get_state(config)

        # 检查是否触发人工介入
        if current_state.next and "human_review" in current_state.next:
            interrupt_data = current_state.tasks[0].interrupts[0].value
            return {
                "status": "need_human_review",
                "message": interrupt_data["message"],
                "draft": interrupt_data["draft"]
            }

        return {"status": "success", "answer": final_state.get("draft_answer", "未生成回答")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/resume")
def resume_chat(q: ResumeQuery):
    """处理人工介入后的恢复"""
    config = {"configurable": {"thread_id": q.thread_id}}
    try:
        # 恢复图执行
        final_state = agent_app.invoke(Command(resume=q.human_input), config=config)
        return {"status": "success", "answer": final_state.get("draft_answer", "未生成回答")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}