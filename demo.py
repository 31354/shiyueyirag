# demo.py —— 多 Agent 科技感看板 (带登录版)
import streamlit as st
import time
import warnings
import os
import logging

# 屏蔽警告，保持终端干净
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from rag import init as rag_init
from multi_agent import app as agent_app

# ============================================================
# 1. 页面基础配置与美化
# ============================================================
st.set_page_config(
    page_title="RAG + Multi-Agent 协作看板",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 美化
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stStatusWidget {
        border: 1px solid #4CAF50;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. 登录认证逻辑
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 欢迎访问多 Agent 协作系统")
    st.info("请先登录以体验 RAG 检索与多 Agent 编排功能。")

    with st.form("login_form"):
        username = st.text_input("用户名 (测试账号: admin)")
        password = st.text_input("密码 (测试密码: 123456)", type="password")
        submit = st.form_submit_button("登 录")

        if submit:
            if username == "admin" and password == "123456":
                st.session_state.authenticated = True
                st.success("登录成功！正在进入系统...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("用户名或密码错误，请重试。")
    st.stop()

# ============================================================
# 3. 登录后的主界面
# ============================================================
st.title("🤖 RAG + Multi-Agent 协作系统")
st.caption("采用 LangGraph 编排：Researcher 查资料 ➔ Analyst 计算 ➔ Writer 撰写 ➔ Reviewer 审核")

with st.sidebar:
    st.header("👤 用户信息")
    st.success("已登录：admin")
    st.divider()

    st.header("📌 系统架构说明")
    st.info(
        "**Researcher**：RAG 知识库检索\n\n"
        "**Analyst**：代码计算/数据处理\n\n"
        "**Writer**：生成最终回答\n\n"
        "**Reviewer**：幻觉审核与重写"
    )
    st.divider()

    if st.button("🚪 退出登录"):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 4. 初始化与历史记录
# ============================================================
@st.cache_resource
def _init():
    rag_init()
    return True


_init()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 5. 核心交互逻辑
# ============================================================
if prompt := st.chat_input("请输入你的问题（例如：对比 FAISS 和 Milvus）..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.status("🧠 多 Agent 正在协作...", expanded=True)
        answer_container = st.empty()

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
            for output in agent_app.stream(initial_state):
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

                    if "draft_answer" in state_update:
                        final_answer = state_update["draft_answer"]

            status_container.update(label="✅ 多 Agent 协作完成", state="complete", expanded=False)

            # 直接显示最终答案（避免高频 DOM 更新触发前端 bug）
            answer_container.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            status_container.update(label="❌ 运行出错", state="error")
            st.error(f"发生错误：{type(e).__name__}: {e}")