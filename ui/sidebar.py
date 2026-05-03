import streamlit as st
from ui.api import api_delete, get_chat_sessions_cached

def render_sidebar():
    with st.sidebar:
        st.title("SmartDoc AI")
        st.caption("Knowledge Graph RAG System")
        st.divider()

        st.subheader("Chat History")
        
        sessions_data = get_chat_sessions_cached()
        if sessions_data and sessions_data.get("sessions"):
            for sess in sessions_data["sessions"][:15]:
                title = sess['title'][:25] + "..." if len(sess['title']) > 25 else sess['title']
                
                col1, col2 = st.columns([5, 1])
                
                if col1.button(f"{title}", key=f"open_{sess['id']}", use_container_width=True):
                    st.session_state["page"] = "Chat"
                    st.session_state["resume_session"] = sess["id"]
                    st.rerun()
                
                if col2.button("✕", key=f"del_{sess['id']}"):
                    api_delete(f"/chat/sessions/{sess['id']}")
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.info("No chat sessions yet.")