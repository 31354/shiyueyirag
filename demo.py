# demo.py —— 多 Agent Streamlit 监控看板
import streamlit as st
import time
from rag import init as rag_init
# 修改：引入我们的多 Agent 工作流
from multi_agent import app as agent_app

st.set_page_config(page_title="RAG + Multi-Agent Demo", page_icon="🤖", layout="wide")
st.title("🤖 RAG + 多 Agent 协作系统")
st.caption("采用 LangGraph 编排：Researcher 查资料 -> Analyst 计算 -> Writer 撰写 -> Reviewer 审核")


@st.cache_resource
def _init():
    rag_init()
    return True


_init()

# 保持对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收输入
if prompt := st.chat_input("请输入你需要分析的复杂问题..."):
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent 处理过程
    with st.chat_message("assistant"):
        # 创建一个状态容器，用于实时展示各个 Agent 的工作进度
        status_container = st.status("🧠 多 Agent 正在协作...", expanded=True)
        answer_container = st.empty()

        # 构建初始状态
        initial_state = {
            "task": prompt,
            "retrieved_docs": "",
            "calculation": "",
            "draft_answer": "",
            "review_passed": False,
            "revision_count": 0
        }

        final_answer = ""

        try:
            # 使用 astream 流式获取每个节点的执行结果
            for output in agent_app.stream(initial_state):
                # output 的键是节点名称，值是节点返回的更新内容
                for node_name, state_update in output.items():
                    if node_name == "researcher":
                        status_container.write("🔍 **[Researcher]** 正在检索知识库并提取关键信息...")
                    elif node_name == "analyst":
                        calc = state_update.get("calculation")
                        if calc:
                            status_container.write(f"🧮 **[Analyst]** 触发计算：`{calc}`")
                        else:
                            status_container.write("🧮 **[Analyst]** 无需数学计算")
                    elif node_name == "writer":
                        status_container.write("✍️ **[Writer]** 正在汇总资料，撰写初稿...")
                    elif node_name == "reviewer":
                        passed = state_update.get("review_passed")
                        if passed:
                            status_container.write("✅ **[Reviewer]** 审核通过，无幻觉产生")
                        else:
                            status_container.write(
                                f"❌ **[Reviewer]** 发现幻觉！打回重写 (第 {state_update.get('revision_count')} 次)")

                    # 记录最终答案
                    if "draft_answer" in state_update:
                        final_answer = state_update["draft_answer"]

            status_container.update(label="✅ 多 Agent 协作完成", state="complete", expanded=False)

            # 打字机效果显示最终答案
            display_text = ""
            for char in final_answer:
                display_text += char
                answer_container.markdown(display_text + "▌")
                time.sleep(0.01)
            answer_container.markdown(final_answer)

            # 存入历史记录
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            status_container.update(label="❌ 运行出错", state="error")
            st.error(f"发生错误：{type(e).__name__}: {e}")