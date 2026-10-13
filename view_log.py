import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r.get("type")

        if t == "user":
            print(f"\n👤 {r['content']}")
        elif t == "turn_start":
            print(f"\n--- 第 {r['turn']} 轮 ---")
        elif t == "tool_call":
            print(f"  🔧 {r['name']}")
            print(f"     参数：{r['args']}")
        elif t == "tool_result":
            print(f"  📥 结果({r['elapsed_sec']}s)：{r['result'][:100]}...")
        elif t == "answer":
            print(f"  🤖 回答({r['elapsed_sec']}s)：{r['content'][:200]}...")
        elif t == "error":
            print(f"  ❌ [{r['where']}] {r['error']}")