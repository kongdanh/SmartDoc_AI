import streamlit as st
from ui.api import api_delete, get_chat_sessions_cached

def render_sidebar():
    with st.sidebar:
        st.title("SmartDoc AI")
        st.caption("Hệ thống tri thức GraphRAG")
        st.divider()

        st.subheader("Lịch sử Chat")
        
        # Nút xóa tất cả
        if st.button("Xóa toàn bộ lịch sử", type="primary", use_container_width=True):
            api_delete("/chat/sessions")
            st.cache_data.clear() # Làm sạch cache để cập nhật giao diện
            st.rerun()

        st.divider()

        # Danh sách lịch sử hiển thị theo chiều dọc
        sessions_data = get_chat_sessions_cached() # Gọi hàm cache
        if sessions_data and sessions_data.get("sessions"):
            for sess in sessions_data["sessions"][:15]:
                title = sess['title'][:25] + "..." if len(sess['title']) > 25 else sess['title']
                
                with st.container(border=True):
                    st.write(f"**{title}**")
                    c1, c2 = st.columns(2)
                    if c1.button("Mở Chat", key=f"open_{sess['id']}", use_container_width=True):
                        st.session_state["page"] = "Chat"
                        st.session_state["resume_session"] = sess["id"]
                        st.rerun()
                    if c2.button("Xóa", key=f"del_{sess['id']}", use_container_width=True):
                        api_delete(f"/chat/sessions/{sess['id']}")
                        st.cache_data.clear() # Làm sạch cache
                        st.rerun()
        else:
            st.info("Chưa có cuộc trò chuyện nào.")