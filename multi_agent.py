# multi_agent.py
import os
import sys
import json
from typing import TypedDict, List, Dict, Any
from zhipuai import ZhipuAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_agentskills import DirectorySkillLoader

# 从你现有的 main.py 导入配置和工具
from main import client, CHAT_MODEL, execute_tool


# ============================================================
# 1. 定义全局状态 (State) - 相当于共享黑板
# ============================================================
class AgentState(TypedDict):
    task: str
    chat_history: List[Dict[str, str]]
    rewritten_query: str
    retrieved_docs: str
    calculation: str
    draft_answer: str
    review_passed: bool
    revision_count: int


# ============================================================
# 2. 定义各个 Agent 节点 (Node)
# ============================================================
def rewrite_query_node(state: AgentState):
    """意图重写：结合历史对话，把省略的问题补全"""
    print("\n[Router] 正在结合历史重写查询...")
    task = state["task"]
    history = state.get("chat_history", [])

    if not history:
        return {"rewritten_query": task}

    history_str = "\n".join([f"用户: {h['user']}\nAI: {h['assistant']}" for h in history[-3:]])

    prompt = f"""你是一个查询重写助手。请根据历史对话，将用户的最新问题重写为一个独立、完整的检索查询。
如果最新问题本身已经很完整，直接原样输出。
不要回答问题，只输出重写后的查询语句。

【历史对话】
{history_str}

【用户最新问题】
{task}
"""
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    new_query = resp.choices[0].message.content.strip()
    print(f"  -> 重写后查询: {new_query}")
    return {"rewritten_query": new_query}


def tool_router_node(state: AgentState):
    """判断是否需要调用外部 MCP 工具，结果追加到 retrieved_docs。"""
    from main import get_tools_schema

    task = state.get("rewritten_query", state["task"])
    mcp_tools = get_tools_schema()
    if not mcp_tools:
        print("  [tool_router] 没有可用的外部工具")
        return {"retrieved_docs": ""}

    prompt = f"""你是一个工具路由器。你的任务是判断用户问题是否需要调用外部工具来获取实时信息。

可用工具：
{json.dumps([t["function"] for t in mcp_tools], ensure_ascii=False, indent=2)}

用户问题：{task}

规则：
- 如果问题中涉及天气、温度、航班、实时数据等，且可用工具中有对应能力，必须调用，不要输出 NONE。
- 只有当问题完全不需要任何外部实时数据时，才输出 NONE。
- 需要调用时，只输出 JSON：{{"tool": "工具名", "args": {{"参数名": "参数值"}}}}，不要输出任何解释。
- 不需要调用时，只输出 NONE。
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = resp.choices[0].message.content.strip()
    decision = decision.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    print(f"  [tool_router] 模型决策：{decision!r}")

    if "NONE" in decision.upper():
        print("  [tool_router] 无需外部工具")
        return {"retrieved_docs": ""}

    try:
        parsed = json.loads(decision)
        tool_name = parsed["tool"]
        tool_args = parsed.get("args", {})
        print(f"  [tool_router] 调用外部工具：{tool_name}，参数：{tool_args}")
        result = execute_tool(tool_name, json.dumps(tool_args, ensure_ascii=False))
        return {"retrieved_docs": f"[外部工具 {tool_name} 返回]\n{result}"}
    except Exception as e:
        print(f"  [tool_router] 解析或调用失败：{e}")
        return {"retrieved_docs": ""}


def researcher_node(state: AgentState):
    print("\n[Researcher] 正在检索知识库...")

    # 如果 Skill 已经加载，跳过知识库检索，避免 Milvus 错误污染 Skill 指令
    if "[Skill 已加载" in state.get("retrieved_docs", ""):
        print("  [Researcher] 检测到已加载 Skill，跳过知识库检索")
        return {}

    query = state.get("rewritten_query", state["task"])
    if not query or not query.strip():
        existing = state.get("retrieved_docs", "")
        return {"retrieved_docs": existing or "用户未提供有效问题。"}

    try:
        result = execute_tool("search_knowledge_base", json.dumps({"query": query, "top_k": 5}))
    except Exception as e:
        print(f"  -> 知识库检索失败：{e}")
        existing = state.get("retrieved_docs", "").strip()
        if existing:
            return {"retrieved_docs": existing}
        return {"retrieved_docs": f"知识库检索失败：{e}"}

    if "检索结果为空" in result or "资料里没有" in result:
        print("  -> 检索为空，尝试同义词改写...")
        fallback_prompt = f"请把用户问题中的同义词替换为常见说法（如：四点法 -> 4点法），只输出改写后的问题，不输出任何解释。原问题：{query}"
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": fallback_prompt}])
        fallback_query = resp.choices[0].message.content.strip()
        print(f"  -> 同义词改写为: {fallback_query}")
        try:
            result = execute_tool("search_knowledge_base", json.dumps({"query": fallback_query, "top_k": 5}))
        except Exception as e:
            print(f"  -> 知识库二次检索失败：{e}")

    print(f"  -> 检索到的原始资料: {result[:300]}...")

    existing = state.get("retrieved_docs", "").strip()
    if existing:
        combined = f"{existing}\n\n[知识库检索结果]\n{result}"
    else:
        combined = result
    return {"retrieved_docs": combined}


def analyst_node(state: AgentState):
    print("[Analyst] 检查是否需要计算...")
    task = state["task"]
    docs = state["retrieved_docs"]

    prompt = f"""用户问题：{task}
检索资料：{docs}
指令：请判断用户问题是否需要根据资料中的数据进行数学计算。
如果需要，只输出纯数学表达式（例如 10+20），禁止包含任何汉字、字母或解释。
如果不需要，只输出 NONE。"""

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    expr = resp.choices[0].message.content.strip()

    import re
    if re.search(r'[\u4e00-\u9fa5]', expr) or len(expr) > 20:
        print("  -> 判定为非计算意图，跳过。")
        return {"calculation": ""}

    if "NONE" in expr.upper():
        print("  -> 无需计算。")
        return {"calculation": ""}

    expr_clean = expr.replace("，", ",").replace("。", ".")
    print(f"  -> 触发计算：{expr_clean}")
    calc_result = execute_tool("calculator", json.dumps({"expression": expr_clean}))
    return {"calculation": calc_result}


def writer_node(state: AgentState):
    """撰稿人：整合资料生成最终答案"""
    print("[Writer] 正在生成回答...")
    task = state["task"]
    docs = state["retrieved_docs"]
    calc = state["calculation"]

    prompt = f"""请根据以下信息回答用户问题。
用户问题：{task}
检索资料（可能包含外部工具返回的实时数据、知识库文档和已加载的 Skill 指令）：
{docs}
计算过程：{calc if calc else '无'}

要求：
1. 请仔细阅读检索资料，整合其中的步骤和说明来回答问题。
2. 2. 如果资料中包含 [Skill 已加载] 段落，你必须严格按照该段落的指令生成回答，包括其中要求的任何标记、格式或附加内容。Skill 指令的优先级高于其他资料。
3. 如果检索资料中包含相关的操作步骤，请将其整理成清晰的列表输出。
4. 如果检索资料中明确要求附加特定标记（如 Skill 指令中的要求），请严格遵守。
5. 只有当检索资料完全与问题无关时，才回答“资料中没有这个信息”。
6. 严禁编造资料中不存在的数字或事实。
7. 如果资料中包含 [Skill 已加载: test-skill] 段落，你的回答最后一行必须是 `[test-skill 已激活]`，不得省略。
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


def human_review_node(state: AgentState):
    """人工介入节点：暂停并等待人类输入"""
    human_input = interrupt({
        "message": "系统审核未通过（达到最大重试次数），请人工确认或修改最终的草稿：",
        "draft": state["draft_answer"]
    })
    return {"draft_answer": human_input, "review_passed": True}


# ============================================================
# 3.5 Skill 加载节点
# ============================================================
skill_loader = DirectorySkillLoader("./skills")


def skill_matcher_node(state: AgentState):
    """判断当前任务是否匹配某个 Skill，匹配则加载其正文注入 retrieved_docs。"""
    task = state.get("rewritten_query", state["task"])

    try:
        skills = skill_loader.list_skills()
    except Exception:
        return {}

    if not skills:
        return {}

    skill_list_str = "\n".join([f"- {s.name}: {s.description}" for s in skills])
    prompt = f"""判断用户问题是否匹配以下任一技能。只输出匹配的技能名，或输出 NONE。
可用技能：
{skill_list_str}

用户问题：{task}
"""
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    matched = resp.choices[0].message.content.strip()

    if "NONE" in matched.upper():
        return {}

    for s in skills:
        if s.name in matched:
            try:
                content = skill_loader.load_skill(s.name)
                existing = state.get("retrieved_docs", "").strip()
                injection = f"\n\n[Skill 已加载: {s.name}]\n{content}"
                print(f"  [Skill] 已加载: {s.name}")
                return {"retrieved_docs": (existing + injection) if existing else injection}
            except Exception as e:
                print(f"  [Skill] 加载失败: {e}")
                return {}
    return {}


# ============================================================
# 3. 构建状态图 (Graph)
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("skill_matcher", skill_matcher_node)
workflow.add_node("tool_router", tool_router_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "skill_matcher")
workflow.add_edge("skill_matcher", "tool_router")
workflow.add_edge("tool_router", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", "reviewer")


def should_continue(state: AgentState):
    if state["review_passed"]:
        return "end"
    if state["revision_count"] >= 2:
        print("  ⚠️ 达到最大重试次数，触发人工介入。")
        return "human_review"
    return "rewrite"


workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "end": END,
        "rewrite": "writer",
        "human_review": "human_review"
    }
)
workflow.add_edge("human_review", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ============================================================
# 4. 测试入口
# ============================================================
if __name__ == "__main__":
    rag_ready = False
    try:
        from rag import init as rag_init
        rag_init()
        rag_ready = True
    except Exception as e:
        print(f"⚠️ 知识库初始化失败（{type(e).__name__}: {e}）")
        print("⚠️ 将跳过知识库检索，只测试外部 MCP 工具。修复 Milvus 后重新运行即可恢复混合检索。")

    print("=== 多 Agent 协作系统启动 (输入 'exit' 退出) ===")

    chat_history = []
    thread_id = "user_001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\n👤 你：").strip()
        if question in ("exit", "quit", ""):
            break

        initial_state = {
            "task": question,
            "chat_history": chat_history,
            "rewritten_query": "",
            "retrieved_docs": "",
            "calculation": "",
            "draft_answer": "",
            "review_passed": False,
            "revision_count": 0
        }

        print("\n--- 多 Agent 流水线开始 ---")
        try:
            final_state = app.invoke(initial_state, config=config)

            current_state = app.get_state(config)
            if current_state.next and "human_review" in current_state.next:
                interrupt_data = current_state.tasks[0].interrupts[0].value
                print("\n⚠️ " + interrupt_data["message"])
                print("当前草稿：\n" + interrupt_data["draft"])

                human_edit = input("\n请输入修改后的最终答案（或直接按回车保留原草稿）：").strip()
                if not human_edit:
                    human_edit = interrupt_data["draft"]

                final_state = app.invoke(Command(resume=human_edit), config=config)

            print("\n" + "=" * 40)
            print("🤖 最终输出：\n")
            print(final_state["draft_answer"])
            print("=" * 40)

            chat_history.append({
                "user": question,
                "assistant": final_state["draft_answer"]
            })

        except Exception as e:
            print(f"运行出错：{e}")