import streamlit as st
import requests
import json
from ui.api import get_domain_names, API_BASE, api_get

def render_chat():
    st.title("Trợ lý AI GraphRAG")
    
    # Thanh công cụ
    col1, col2 = st.columns([3, 1])
    domain_names = get_domain_names(only_ready=True)
    domain = col1.selectbox("Chọn nguồn dữ liệu (Domain):", domain_names if domain_names else ["Chưa có"])
    
    if col2.button("Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.pop("chat_session_id", None)
        st.session_state["messages"] = []
        st.rerun()

    st.divider()

    # Phục hồi lịch sử chat
    if "resume_session" in st.session_state:
        sess_id = st.session_state.pop("resume_session")
        sess_data = api_get(f"/chat/sessions/{sess_id}")
        if sess_data:
            st.session_state["chat_session_id"] = sess_data["id"]
            st.session_state["messages"] = [{"role": m["role"], "content": m["content"]} for m in sess_data.get("messages", [])]

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Khung Chat giống Messenger
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Khung nhập liệu (Luôn bám đáy màn hình)
    if domain and domain != "Chưa có":
        if prompt := st.chat_input("Hỏi bất cứ điều gì về tài liệu..."):
            
            # 1. In câu hỏi của user
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
                
                # 2. Xử lý câu trả lời của AI (Streaming)
                with st.chat_message("assistant"):
                    session_id = st.session_state.get("chat_session_id")
                    payload = {"domain": domain, "message": prompt}
                    if session_id:
                        payload["session_id"] = session_id

                    try:
                        res = requests.post(f"{API_BASE}/chat", json=payload, stream=True, timeout=120)
                        if res.status_code == 200:
                            def stream_data():
                                for line in res.iter_lines():
                                    if line:
                                        decoded = line.decode("utf-8")
                                        if decoded.startswith("data: "):
                                            try:
                                                data = json.loads(decoded[6:])
                                                if data.get("type") == "start":
                                                    st.session_state["chat_session_id"] = data.get("session_id")
                                                elif data.get("type") == "token":
                                                    yield data.get("content", "")
                                            except (json.JSONDecodeError, KeyError, IndexError): pass
                            # Tạo hiệu ứng gõ phím
                            full_response = st.write_stream(stream_data())
                            st.session_state["messages"].append({"role": "assistant", "content": full_response})
                        else:
                            st.error(f"Lỗi Server: {res.status_code}")
                    except Exception as e:
                        st.error(f"Lỗi kết nối: {e}")