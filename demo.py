import streamlit as st

from rag import init as rag_init
from main import SYSTEM_PROMPT, run_agent_stream

st.set_page_config(page_title="RAG + Agent Demo", page_icon="🤖")
st.title("🤖 RAG + Agent Demo")
st.caption("问关于知识库的问题，或让我做计算")


@st.cache_resource
def _init():
    rag_init()
    return True


_init()

if "display" not in st.session_state:
    st.session_state.display = []
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]

# 显示历史
for msg in st.session_state.display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if q := st.chat_input("输入你的问题…"):
    st.session_state.display.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    st.session_state.history.append({"role": "user", "content": q})

    with st.chat_message("assistant"):
        status = st.empty()       # 显示过程
        answer_box = st.empty()   # 显示答案

        # 收集事件（不用闭包改外部变量，避免黑屏）
        events = []


        def on_event(etype, data):
            events.append((etype, data))

        try:
            ans = run_agent_stream(st.session_state.history, on_event=on_event)
        except Exception as e:
            status.empty()
            ans = f"出错：{type(e).__name__}: {e}"
            answer_box.error(ans)
            st.session_state.display.append({"role": "assistant", "content": ans})
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.stop()

        # 重放过程事件
        for etype, data in events:
            if etype == "turn":
                status.info(f"🔄 第 {data['turn']} 轮")
            elif etype == "tool_call":
                status.info(f"🔧 调用工具：{data['name']}")
            elif etype == "tool_result":
                status.info(f"✅ {data['name']} 完成")
        status.empty()

        # 假流式：逐字打印最终答案
        full = ""
        for ch in (ans or ""):
            full += ch
            answer_box.markdown(full + "▌")
        answer_box.markdown(full or "（无回答）")

    st.session_state.display.append({"role": "assistant", "content": ans})
    st.session_state.history.append({"role": "assistant", "content": ans})