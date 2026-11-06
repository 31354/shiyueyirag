import streamlit as st
import requests
import uuid

# 强制使用 127.0.0.1，避免 localhost 解析问题
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="智能运维 Agent", page_icon="🤖")
st.title("🤖 工业机器人智能运维 Agent")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框
if prompt := st.chat_input("请输入您的问题（例如：怎么用XYZ 4点法测量工具？）"):
    # 1. 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 请求后端并显示 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在翻阅手册并思考（约需 10-20 秒）..."):

            # 🚨 终端探测器：如果 Streamlit 黑窗口没打印这句，说明前端代码没跑或没保存！
            print(f"\n[前端] 正在发送请求到 {API_URL}/chat | 问题: {prompt}")

            try:
                # 发送 POST 请求
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=120  # 设置 2 分钟超时，防止多 Agent 跑得慢
                )

                # 🚨 终端探测器：看看后端返回了什么状态码
                print(f"[前端] 收到后端响应，状态码: {resp.status_code}")

                if resp.status_code == 200:
                    data = resp.json()

                    # 处理人工介入的情况（如果后端触发了 human_review）
                    if data.get("status") == "need_human_review":
                        st.warning(data.get("message", "系统触发人工介入"))
                        st.info(f"**AI 草稿：**\n\n{data.get('draft', '')}")
                        st.stop()  # 暂停，等待人类处理

                    # 正常返回
                    answer = data.get("answer", "未生成有效回答")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                else:
                    st.error(f"后端报错 (HTTP {resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ 无法连接到后端！请确认 FastAPI 已在 {API_URL} 启动。")
            except requests.exceptions.Timeout:
                st.error("❌ 请求超时！多 Agent 思考时间过长，请重试。")
            except Exception as e:
                st.error(f"❌ 发生未知错误: {e}")