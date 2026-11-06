import os
import sys
import json
import time
import warnings
import operator as _op
from zhipuai import ZhipuAI
import asyncio
from pathlib import Path
from fastmcp import Client

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

# ============================================================
# 0. MCP 工具发现（惰性，避免导入时触发事件循环冲突）
# ============================================================
MCP_SERVER_PATH = Path("D:/shiyueyirag/moniwaibugongju.py")


async def _fetch_mcp_tools():
    async with Client(MCP_SERVER_PATH) as client:
        mcp_tools = await client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def _run_async(coro):
    """在同步上下文里安全地跑一个协程，兼容已有事件循环的情况。"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def get_tools_schema():
    """在需要时拉取 MCP 工具列表并转换成 Function Calling schema。"""
    try:
        return _run_async(_fetch_mcp_tools())
    except Exception as e:
        print(f"  [警告] 拉取 MCP 工具失败：{type(e).__name__}: {e}")
        return []

async def get_langchain_mcp_tools():
    """用 LangChain 适配器加载 MCP 工具（返回 BaseTool 列表）。"""
    client = MultiServerMCPClient({
        "moniwaibugongju": {
            "command": "python",
            "args": [str(MCP_SERVER_PATH)],
            "transport": "stdio",
        }
    })
    return await client.get_tools()

def _call_mcp_tool(name, args):
    async def _run():
        async with Client(MCP_SERVER_PATH) as client:
            result = await client.call_tool(name, args)
            return result.data

    try:
        return _run_async(_run())
    except Exception as e:
        return f"错误：MCP 工具 {name} 调用失败，{type(e).__name__}: {e}"


# ============================================================
# 1. 配置与大模型 Client
# ============================================================
client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
if not client.api_key:
    raise SystemExit("请先设置环境变量 ZHIPU_API_KEY")

CHAT_MODEL = "glm-4-flash"

# ============================================================
# 2. 安全计算器 (AST 沙箱)
# ============================================================
_ALLOWED_OPS = {
    __import__('ast').Add: _op.add, __import__('ast').Sub: _op.sub,
    __import__('ast').Mult: _op.mul, __import__('ast').Div: _op.truediv,
    __import__('ast').Pow: _op.pow, __import__('ast').Mod: _op.mod,
    __import__('ast').FloorDiv: _op.floordiv, __import__('ast').USub: _op.neg,
    __import__('ast').UAdd: _op.pos,
}


def _safe_eval(node):
    import ast
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
    import ast
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{type(e).__name__}: {e}"

# ============================================================
# 3. 通用工具执行器
# ============================================================
def execute_tool(name, args_json):
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return f"错误：参数不是合法 JSON，{e}"

    if name == "search_knowledge_base":
        from rag import retrieve_for_agent
        docs = retrieve_for_agent(args.get("query", ""), top_k=args.get("top_k", 5))
        if not docs:
            return "检索结果为空。请如实告诉用户：资料里没有这个信息，禁止编造。"
        result = "\n\n---\n\n".join(docs)
        if len(result) > 1500:
            result = result[:1500] + "\n\n（结果已截断）"
        return result

    if name == "calculator":
        return calculator(args.get("expression", ""))

    # 本地没匹配到，尝试走 MCP
    return _call_mcp_tool(name, args)

# ============================================================
# 4. 通用 LLM 调用函数 (带重试)
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