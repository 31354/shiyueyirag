# multi_agent.py
import os
import sys
import json
from typing import TypedDict, List, Dict, Any
from zhipuai import ZhipuAI

# 从你现有的 main.py 导入配置和工具
from main import client, CHAT_MODEL, execute_tool

from langgraph.graph import StateGraph, END


# ============================================================
# 1. 定义全局状态 (State) - 相当于共享黑板
# ============================================================
class AgentState(TypedDict):
    task: str  # 用户的原始问题
    retrieved_docs: str  # Researcher 查到的资料
    calculation: str  # Analyst 计算的结果
    draft_answer: str  # 最终生成的答案
    review_passed: bool  # 审核是否通过
    revision_count: int  # 修改次数（防止死循环）


# ============================================================
# 2. 定义各个 Agent 节点 (Node)
# ============================================================
def researcher_node(state: AgentState):
    """研究员：负责调用 RAG 检索"""
    print("\n[Researcher] 正在检索知识库...")
    query = state["task"]

    if not query or not query.strip():
        return {"retrieved_docs": "用户未提供有效问题。"}

    # 复用你的工具执行函数
    result = execute_tool("search_knowledge_base", json.dumps({"query": query, "top_k": 3}))

    return {"retrieved_docs": result}


def analyst_node(state: AgentState):
    """分析师：如果问题有数字计算，这里调用大模型提取表达式并计算"""
    print("[Analyst] 检查是否需要计算...")
    task = state["task"]
    docs = state["retrieved_docs"]

    # 简单策略：让模型判断是否需要计算
    prompt = f"根据用户问题：{task}\n和资料：{docs}\n如果需要数学计算，请直接输出计算表达式（如 10+20）；如果不需要，输出 'NONE'。"
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    expr = resp.choices[0].message.content.strip()

    if expr != "NONE" and expr:
        print(f"  -> 触发计算：{expr}")
        calc_result = execute_tool("calculator", json.dumps({"expression": expr}))
        return {"calculation": calc_result}

    return {"calculation": ""}


def writer_node(state: AgentState):
    """撰稿人：整合资料生成最终答案"""
    print("[Writer] 正在生成回答...")
    task = state["task"]
    docs = state["retrieved_docs"]
    calc = state["calculation"]

    prompt = f"""请根据以下信息回答用户问题。
用户问题：{task}
检索资料：{docs}
计算过程：{calc if calc else '无'}
要求：
1. 只能基于资料回答，禁止编造。
2. 如果资料不足，请直接说明不知道。
3. 输出要求自然流畅。
"""
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return {"draft_answer": resp.choices[0].message.content}


def reviewer_node(state: AgentState):
    """审核员：检查草稿是否忠于资料，有没有幻觉"""
    print("[Reviewer] 正在审核回答真实性...")
    task = state["task"]
    docs = state["retrieved_docs"]
    draft = state["draft_answer"]
    count = state["revision_count"]

    prompt = f"""你是严格的审核员。
用户问题：{task}
提供的资料：{docs}
生成的草稿：{draft}

请检查草稿中是否有资料里**没有**的数字、事实或结论。
如果有幻觉（资料中不存在），输出 'REJECT: 具体的修改意见'。
如果完全基于资料，输出 'PASS'。
"""
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    review_result = resp.choices[0].message.content.strip()

    if review_result.startswith("PASS"):
        print("  ✅ 审核通过")
        return {"review_passed": True}
    else:
        print(f"  ❌ 审核拒绝 (第 {count + 1} 次)")
        return {"review_passed": False, "revision_count": count + 1}


# ============================================================
# 3. 构建状态图 (Graph)
# ============================================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

# 定义顺序边
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", "reviewer")


# 定义条件边（审核后是 END 还是回到 Writer）
def should_continue(state: AgentState):
    if state["review_passed"]:
        return "end"
    if state["revision_count"] >= 2:  # 最多重试 2 次，防止死循环
        print("  ⚠️ 达到最大重试次数，强制结束。")
        return "end"
    return "rewrite"


workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "end": END,
        "rewrite": "writer"
    }
)

# 编译图
app = workflow.compile()

# ============================================================
# 4. 测试入口
# ============================================================
if __name__ == "__main__":
    from rag import init as rag_init

    rag_init()  # 初始化你的 RAG

    print("=== 多 Agent 协作系统启动 (输入 'exit' 退出) ===")

    while True:
        question = input("\n👤 你：").strip()
        if question in ("exit", "quit", ""):
            break

        initial_state = {
            "task": question,
            "retrieved_docs": "",
            "calculation": "",
            "draft_answer": "",
            "review_passed": False,
            "revision_count": 0
        }

        print("\n--- 多 Agent 流水线开始 ---")
        try:
            final_state = app.invoke(initial_state)
            print("\n" + "=" * 40)
            print("🤖 最终输出：\n")
            print(final_state["draft_answer"])
            print("=" * 40)
        except Exception as e:
            print(f"运行出错：{e}")