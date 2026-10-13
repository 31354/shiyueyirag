# tools.py
from tools_registry import register_tool
# 或者你之前用的 TOOL_REGISTRY 方式，保持不变

@register_tool(
    name="search_knowledge_base",
    description=(
        "从内部知识库检索相关文档片段。当问题涉及公司政策、产品文档、"
        "技术资料等内部知识时使用。返回最相关的几个片段，包含来源。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词或问题，尽量具体",
            },
            "top_k": {
                "type": "integer",
                "description": "返回片段数量，默认 5",
            },
        },
        "required": ["query"],
    },
)
def search_knowledge_base(query: str, top_k: int = 5):
    from rag import retrieve
    docs = retrieve(query, top_k=top_k)
    if not docs:
        return "未检索到相关文档"
    return "\n\n---\n\n".join(docs)