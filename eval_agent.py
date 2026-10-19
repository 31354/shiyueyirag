# eval_agent.py —— 多 Agent 端到端行为评测
import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from rag import init as rag_init
# 引入你的多 Agent 状态图
from multi_agent import app as agent_app

rag_init()

# (问题, 期望回答包含的关键词)
cases = [
    ("什么是向量数据库？", "向量"),
    ("123 * 456 等于多少？", "56088"),
    ("Recall@5 是多少？把它乘以 100。", "85"),
    ("知识库里有讲火星移民吗？", "没有"),
    ("Chroma 和 Milvus 有什么区别？", "Chroma"),
    ("(120-80)/80*100 等于多少？", "50"),
]

passed = 0
print("开始多 Agent 端到端评测...")

for q, expect_kw in cases:
    print(f"\n{'=' * 60}\n问题：{q}")

    initial_state = {
        "task": q,
        "retrieved_docs": "",
        "calculation": "",
        "draft_answer": "",
        "review_passed": False,
        "revision_count": 0
    }

    try:
        start_time = time.time()
        # 运行 LangGraph 工作流
        final_state = agent_app.invoke(initial_state)
        end_time = time.time()

        ans = final_state.get("draft_answer", "")
        cost_time = end_time - start_time

    except Exception as e:
        print(f"  ❌ 异常：{type(e).__name__}: {e}")
        continue

    print(f"回答：{ans}")
    print(f"耗时：{cost_time:.2f} 秒")

    ok = expect_kw in (ans or "")
    if ok:
        passed += 1
    print(f"  {'✅' if ok else '❌'} 期望包含「{expect_kw}」")

print(f"\n{'=' * 60}")
print(f"多 Agent 评测完成：通过 {passed}/{len(cases)}")