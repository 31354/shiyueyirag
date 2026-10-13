import os
import sys
import ast
import json
import time
import warnings
import operator as _op
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    # Linux 容器里可能没有 en_US.UTF-8，退回到默认 locale
    pass
from zhipuai import ZhipuAI

from logger import AgentLogger
from rag import retrieve_for_agent, init as rag_init

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

# ============================================================
# 配置
# ============================================================
client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
if not client.api_key:
    raise SystemExit("请先设置环境变量 ZHIPU_API_KEY")

CHAT_MODEL = "glm-4-air-250414"
MAX_TOOL_RESULT_CHARS = 1500
MAX_TURNS = 5

# ============================================================
# calculator 工具
# ============================================================
_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{type(e).__name__}: {e}"


# ============================================================
# 工具定义
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索内部知识库，获取公司/私有文档信息。"
                "涉及内部资料时必须先用此工具，禁止凭记忆回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题，尽量具体"},
                    "top_k": {"type": "integer", "description": "返回片段数量，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "执行数学计算。当需要对数字做加减乘除、百分比、幂运算时使用。"
                "expression 必须是合法数学表达式，如 '(120-80)/80*100'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，仅支持 + - * / // % ** 和括号、数字",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_tool(name, args_json):
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return f"错误：参数不是合法 JSON，{e}"

    if name == "search_knowledge_base":
        docs = retrieve_for_agent(args.get("query", ""), top_k=args.get("top_k", 5))
        if not docs:
            return "检索结果为空。请如实告诉用户：资料里没有这个信息，禁止编造。"
        result = "\n\n---\n\n".join(docs)
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "\n\n（结果已截断）"
        return result

    if name == "calculator":
        return calculator(args.get("expression", ""))

    return f"错误：未知工具 {name}"


# ============================================================
# 模型调用
# ============================================================
def chat(messages, tools=None, tool_choice="auto", max_retries=3):
    kwargs = {"model": CHAT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [重试] 第 {attempt+1} 次失败：{type(e).__name__}: {e}，{wait}s 后重试")
            time.sleep(wait)

    raise RuntimeError(f"模型调用失败（已重试 {max_retries} 次）：{last_error}")


# ============================================================
# Agent 循环
# ============================================================
SYSTEM_PROMPT = """你是一个严谨的助手，只能基于工具返回的资料回答问题。

可用工具：
- search_knowledge_base：检索内部知识库。
- calculator：执行数学计算，如加减乘除、百分比、幂运算。

规则：
1.  只要问题可能涉及内部资料，即使你觉得之前看过类似内容，也必须重新调用 search_knowledge_base。禁止根据"上一轮已经查过"就跳过检索。
2. 回答只能使用检索结果中明确出现的信息。不得添加资料里没有的事实、数字、来源。
3. 如果检索结果里没有所需信息，直接回答"资料里没有这个信息"，不要猜测、不要编造。
4. 涉及数字计算时，必须调用 calculator，不要心算。
5. 回答时标注依据的来源文件名。
6. 不要用相同或几乎相同的参数反复调用同一个工具。
7. 信息足够后直接用自然语言回答，不要再调用工具。
"""


def run_agent_with_history(history, verbose=True, logger: AgentLogger = None):
    turn_start = time.time()
    seen_calls = set()

    for turn in range(MAX_TURNS):
        if turn == 0:
            tool_choice = "required"
        else:
            tool_choice = "auto"
        if logger:
            logger.log_turn_start(turn + 1, tool_choice)

        try:
            msg = chat(history, tools=TOOLS, tool_choice=tool_choice)
        except Exception as e:
            if logger:
                logger.log_error("chat", f"{type(e).__name__}: {e}")
            raise

        history.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            if verbose:
                print(f"\n[第 {turn+1} 轮] 模型直接回答")
            if logger:
                logger.log_answer(msg.content or "", time.time() - turn_start)
            return msg.content

        if verbose:
            print(f"\n[第 {turn+1} 轮] 模型请求 {len(msg.tool_calls)} 个工具调用")

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            sig = (name, args)

            if logger:
                logger.log_tool_call(name, args)

            if sig in seen_calls:
                result = ("你已经用完全相同的参数调用过这个工具，结果不会变。"
                          "请换关键词，或基于已有信息作答；如果确实查不到，就如实说明。")
                if verbose:
                    print(f"  ⚠️ 重复调用 {name}({args})，已拦截")
                if logger:
                    logger.log_tool_result(name, result, 0.0)
            else:
                seen_calls.add(sig)
                if verbose:
                    print(f"  → 调用 {name}({args})")
                t0 = time.time()
                result = execute_tool(name, args)
                elapsed = time.time() - t0
                if logger:
                    logger.log_tool_result(name, result, elapsed)
                if verbose:
                    preview = result[:120].replace("\n", " ")
                    print(f"  ← 结果({elapsed:.2f}s): {preview}{'...' if len(result) > 120 else ''}")

            if name == "search_knowledge_base" and result.startswith("检索结果为空"):
                if verbose:
                    print("  ⚠️ 检索为空，直接返回固定话术")
                if logger:
                    logger.log_answer("资料里没有这个信息。", time.time() - turn_start)
                return "资料里没有这个信息。"

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    if logger:
        logger.log_error("run_agent", f"超过最大轮数 {MAX_TURNS}")

    history.append({
        "role": "user",
        "content": "已达到工具调用上限。请基于目前已有的信息回答，如果信息不足就如实说明无法回答。",
    })
    try:
        final = chat(history)
        return final.content or "抱歉，信息不足，无法回答。"
    except Exception as e:
        if logger:
            logger.log_error("final_answer", f"{type(e).__name__}: {e}")
        return "抱歉，任务步骤过多，已停止。"

# ============================================================
# 流式 / 事件回调版 Agent（供 Streamlit Demo 用）
# ============================================================
def chat_stream(messages, tools=None, tool_choice="auto"):
    """流式调用，yield 增量内容"""
    kwargs = {"model": CHAT_MODEL, "messages": messages, "stream": True}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    resp = client.chat.completions.create(**kwargs)
    for chunk in resp:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        except Exception:
            continue


def run_agent_stream(history, on_event=None, verbose=False, logger=None):
    """
    带事件回调的 Agent。
    on_event(event_type, data) 回调：
      - "turn"       -> {"turn": n}
      - "tool_call"  -> {"name": ..., "args": ...}
      - "tool_result"-> {"name": ..., "preview": ...}
      - "answer"     -> {"content": ...}
      - "error"      -> {"message": ...}
    """
    turn_start = time.time()
    seen_calls = set()

    def emit(etype, **kw):
        if on_event:
            on_event(etype, kw)

    for turn in range(MAX_TURNS):
        tool_choice = "auto"
        if turn == 0:
            tool_choice = {"type": "function", "function": {"name": "search_knowledge_base"}}

        emit("turn", turn=turn + 1)

        try:
            msg = chat(history, tools=TOOLS, tool_choice=tool_choice)
        except Exception as e:
            emit("error", message=f"{type(e).__name__}: {e}")
            raise

        history.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            emit("answer", content=msg.content or "")
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            sig = (name, args)

            emit("tool_call", name=name, args=args)

            if sig in seen_calls:
                result = "你已经用完全相同的参数调用过这个工具，结果不会变。请换关键词或基于已有信息作答。"
            else:
                seen_calls.add(sig)
                result = execute_tool(name, args)

            if name == "search_knowledge_base" and result.startswith("检索结果为空"):
                emit("answer", content="资料里没有这个信息。")
                return "资料里没有这个信息。"

            emit("tool_result", name=name, preview=result[:120].replace("\n", " "))

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    history.append({
        "role": "user",
        "content": "已达到工具调用上限。请基于目前已有的信息回答，如果信息不足就如实说明。",
    })
    try:
        final = chat(history)
        emit("answer", content=final.content or "")
        return final.content
    except Exception as e:
        emit("error", message=str(e))
        return "抱歉，任务步骤过多，已停止。"

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    rag_init()

    logger = AgentLogger()
    print(f"本次会话日志：{logger.session_file}")

    MAX_DIALOGUE_TURNS = 8
    MAX_HISTORY_MESSAGES = 8
    dialogue_count = 0

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        if dialogue_count >= MAX_DIALOGUE_TURNS:
            print(f"\n已达最大对话轮数（{MAX_DIALOGUE_TURNS}），退出。")
            break

        q = input("\n你：").strip()
        if q in ("exit", "quit", ""):
            break

        dialogue_count += 1
        logger.log_user(q)
        history.append({"role": "user", "content": q})

        try:
            answer = run_agent_with_history(history, logger=logger)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            break

        print(f"\n助手：{answer}")

        if len(history) > MAX_HISTORY_MESSAGES + 1:
            history = [history[0]] + history[-MAX_HISTORY_MESSAGES:]