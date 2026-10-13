# eval_agent.py —— Agent 端到端行为评测
# 运行：python eval_agent.py
import os
import sys
import io

# 强制 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.environ.setdefault("PYTHONUTF8", "1")

from rag import init as rag_init
from main import run_agent_with_history, SYSTEM_PROMPT

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
for q, expect_kw in cases:
    print(f"\n{'='*60}\n问题：{q}")
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    try:
        ans = run_agent_with_history(history, verbose=False)
    except Exception as e:
        print(f"  ❌ 异常：{type(e).__name__}: {e}")
        continue
    print(f"回答：{ans}")
    ok = expect_kw in (ans or "")
    if ok:
        passed += 1
    print(f"  {'✅' if ok else '❌'} 期望包含「{expect_kw}」")

print(f"\n{'='*60}")
print(f"通过 {passed}/{len(cases)}")