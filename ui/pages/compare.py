import streamlit as st
from ui.api import get_domain_names, api_post

def render_compare():
    st.title("⚖️ So Sánh Standard RAG vs GraphRAG")
    
    all_domain_names = get_domain_names()

    # Thanh điều khiển
    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        c_domain = c1.selectbox("Chọn Domain", all_domain_names if all_domain_names else ["Chưa có dữ liệu"])
        if c2.button("🔄 Lọc mới", use_container_width=True):
            st.session_state["compare_messages"] = []
            st.rerun()

    if "compare_messages" not in st.session_state:
        st.session_state["compare_messages"] = []

    st.write("")

    # 2 Cột hiển thị Chat chuẩn mực
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Standard RAG")
        st.divider()
    with col2:
        st.subheader("GraphRAG")
        st.divider()

    # Render lịch sử
    for msg in st.session_state["compare_messages"]:
        if msg["role"] == "user":
            with col1: st.chat_message("user").write(msg["content"])
            with col2: st.chat_message("user").write(msg["content"])
        else:
            with col1: st.chat_message("assistant").write(msg["standard_rag"])
            with col2: st.chat_message("assistant").write(msg["graph_rag"])

    # Xử lý nhập liệu
    if c_domain and c_domain != "Chưa có dữ liệu":
        if prompt := st.chat_input("Nhập câu hỏi để so sánh 2 phương pháp..."):
            with col1: st.chat_message("user").write(prompt)
            with col2: st.chat_message("user").write(prompt)

            with col1:
                std_status = st.empty()
                std_status.info("⏳ Đang quét Vector Search...")
            with col2:
                graph_status = st.empty()
                graph_status.info("🧭 Đang duyệt Knowledge Graph...")

            res = api_post("/api/compare-rag", json_data={"query": prompt.strip(), "domain": c_domain})

            if res and res.status_code == 200:
                data = res.json()
                std_ans = data.get("standard_rag_answer", "Không tìm thấy dữ liệu.")
                graph_ans = data.get("graph_rag_answer", "Không tìm thấy dữ liệu.")
                
                std_status.empty()
                graph_status.empty()

                with col1: st.chat_message("assistant").write(std_ans)
                with col2: st.chat_message("assistant").write(graph_ans)

                st.session_state["compare_messages"].extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "standard_rag": std_ans, "graph_rag": graph_ans}
                ])
            else:
                st.error("❌ Lỗi gọi API.")