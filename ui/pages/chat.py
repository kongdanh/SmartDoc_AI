import streamlit as st
import requests
from ui.api import get_domain_names, API_BASE

def render_chat():
    st.title("Chat")
    
    # Header with domain selector
    col1, col2 = st.columns([4, 1])
    
    domain_names = get_domain_names(only_ready=True)
    domain = col1.selectbox(
        "Domain:",
        domain_names if domain_names else ["No domain"],
        key="chat_domain"
    )
    
    if col2.button("New Chat"):
        st.session_state.pop("chat_session_id", None)
        st.session_state["chat_msgs"] = []
        st.rerun()
    
    # Initialize chat messages
    if "chat_msgs" not in st.session_state:
        st.session_state["chat_msgs"] = []
    
    if "resume_session" in st.session_state:
        sess_id = st.session_state.pop("resume_session")
        from ui.api import api_get
        sess_data = api_get(f"/chat/sessions/{sess_id}")
        if sess_data:
            st.session_state["chat_session_id"] = sess_data["id"]
            st.session_state["chat_msgs"] = [
                {"role": m["role"], "content": m["content"]}
                for m in sess_data.get("messages", [])
            ]
    
    if not domain or domain == "No domain":
        st.warning("Please select a domain")
        return
    
    # Chat messages container
    st.divider()
    msg_container = st.container()
    
    with msg_container:
        if not st.session_state["chat_msgs"]:
            st.info("No messages. Start a conversation!")
        else:
            for msg in st.session_state["chat_msgs"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    
    # Chat input at bottom
    st.divider()
    prompt = st.chat_input("Type a message...")
    
    if prompt:
        st.session_state["chat_msgs"].append({"role": "user", "content": prompt})
        
        session_id = st.session_state.get("chat_session_id")
        payload = {"domain": domain, "message": prompt}
        if session_id:
            payload["session_id"] = session_id
        
        with st.spinner("Waiting for response..."):
            try:
                res = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("response") or data.get("content") or str(data)
                    
                    if "session_id" in data:
                        st.session_state["chat_session_id"] = data["session_id"]
                    
                    st.session_state["chat_msgs"].append({"role": "assistant", "content": answer})
                else:
                    st.session_state["chat_msgs"].append({"role": "assistant", "content": f"Error: {res.status_code}"})
            except Exception as e:
                st.session_state["chat_msgs"].append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        st.rerun()