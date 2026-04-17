import streamlit as st
import requests
import time
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config # Cần: pip install streamlit-agraph

# Cấu hình trang
st.set_page_config(page_title="KnowledgeDB - GraphRAG Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- HẰNG SỐ & CẤU HÌNH ---
API_BASE = "http://localhost:8001"
DEFAULT_DOMAIN = "test1"

# --- STYLE CSS (Giữ phong cách tối của dashboard.html) ---
st.markdown("""
    <style>
    .main { background-color: #0f1117; color: #e4e6f0; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .stMetric { background-color: #1a1d27; padding: 15px; border-radius: 10px; border: 1px solid #2e3348; }
    .chat-bubble { padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚀 KnowledgeDB")
    st.caption("GraphRAG Admin Dashboard")
    page = st.radio("Menu", ["Overview", "Chat", "Query", "Knowledge Graph", "Upload Files", "Compare RAG"])
    
    st.divider()
    st.info("Status: Backend Connected" if requests.get(f"{API_BASE}/status").status_code == 200 else "Status: Backend Offline")

# --- HELPER FUNCTIONS ---
def get_domains():
    try:
        res = requests.get(f"{API_BASE}/domains")
        return res.json()["domains"]
    except: return []

# --- 1. OVERVIEW PAGE ---
if page == "Overview":
    st.header("📊 System Overview")
    domains_data = get_domains()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Domains", len(domains_data))
    col2.metric("Ready Domains", len([d for d in domains_data if d['ready']]))
    col3.metric("Indexed Files", sum([d['indexed_files'] for d in domains_data]))

    st.subheader("Domain Management")
    for d in domains_data:
        with st.expander(f"📁 {d['name']} - {'✅ Ready' if d['ready'] else '⏳ Indexing'}"):
            st.write(f"Indexed Files: {d['indexed_files']}")
            if st.button(f"Re-index {d['name']}", key=f"re_{d['name']}"):
                requests.post(f"{API_BASE}/reindex/{d['name']}")
                st.toast(f"Re-indexing {d['name']} started!")

# --- 2. CHAT PAGE ---
elif page == "Chat":
    st.header("💬 AI Chat Assistant")
    domain = st.selectbox("Select Domain", [d['name'] for d in get_domains()])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi gì đó về kiến thức của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                import json
                # Bật chế độ stream=True để hứng luồng dữ liệu SSE
                res = requests.post(f"{API_BASE}/chat", json={"domain": domain, "message": prompt}, stream=True)
                
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:] # Cắt bỏ chữ "data: " ở đầu
                                try:
                                    payload = json.loads(data_str)
                                    if payload.get("type") == "token":
                                        full_response += payload.get("content", "")
                                        # Hiển thị từng chữ với hiệu ứng con trỏ nhấp nháy
                                        message_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    pass
                    
                    # Khi stream xong, in ra kết quả hoàn chỉnh
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    message_placeholder.error(f"Lỗi API: {res.status_code}")
            except Exception as e:
                message_placeholder.error(f"Lỗi kết nối Backend: {e}")

# --- 3. QUERY PAGE ---
elif page == "Query":
    st.header("🔍 Advanced Query")
    col1, col2, col3 = st.columns([2, 1, 1])
    domain = col1.selectbox("Domain", [d['name'] for d in get_domains()])
    method = col2.selectbox("Method", ["search", "local", "global", "drift"])
    level = col3.slider("Community Level", 1, 5, 2)
    
    query_text = st.text_area("Query", placeholder="Nhập câu hỏi truy vấn sâu...")
    
    if st.button("Execute Search", type="primary"):
        with st.spinner("Searching..."):
            endpoint = f"/query/{method}"
            body = {"domain": domain, "query": query_text, "community_level": level, "response_type": "Multiple Paragraphs"}
            if method == "search": body = {"domain": domain, "query": query_text}
            
            res = requests.post(f"{API_BASE}{endpoint}", json=body)
            if res.status_code == 200:
                st.success("Result:")
                st.markdown(res.json().get("response", "No response content"))
            else:
                st.error("Query failed")

# --- 4. KNOWLEDGE GRAPH PAGE ---
elif page == "Knowledge Graph":
    st.header("🕸️ Visual Knowledge Graph")
    domain = st.selectbox("Select Domain to Visualize", [d['name'] for d in get_domains()])
    
    if st.button("Load Graph"):
        res = requests.get(f"{API_BASE}/graph/{domain}")
        if res.status_code == 200:
            data = res.json()
            nodes = [Node(id=n['id'], label=n['id'], size=10+(n['degree']*2), color="#6c8aff") for n in data['nodes']]
            edges = [Edge(source=e['source'], target=e['target']) for e in data['edges']]
            config = Config(width=1000, height=600, directed=True, nodeHighlightBehavior=True)
            agraph(nodes=nodes, edges=edges, config=config)

# --- 5. UPLOAD PAGE ---
elif page == "Upload Files":
    st.header("📤 Data Ingestion")
    new_domain = st.text_input("Target Domain Name", placeholder="e.g. tai_lieu_moi")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    
    if st.button("Upload and Process"):
        if new_domain and uploaded_files:
            files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
            res = requests.post(f"{API_BASE}/domains/{new_domain}/upload", files=files)
            if res.status_code == 200:
                st.success(f"Uploaded to {new_domain}. Starting RAG & GraphRAG indexing...")
                requests.post(f"{API_BASE}/reindex/{new_domain}")
            else:
                st.error("Upload failed")

# --- 6. COMPARE RAG PAGE (Tính năng cải tiến) ---
elif page == "Compare RAG":
    st.header("⚖️ RAG vs GraphRAG Comparison")
    domain = st.selectbox("Select Domain", [d['name'] for d in get_domains()], key="comp_dom")
    comp_query = st.text_input("Nhập câu hỏi để so sánh hiệu quả:")
    
    if st.button("Run Comparison"):
        col_left, col_right = st.columns(2)
        with st.spinner("Đang chạy đối soát..."):
            res = requests.post(f"{API_BASE}/api/compare-rag", json={"query": comp_query, "domain": domain})
            if res.status_code == 200:
                data = res.json()
                with col_left:
                    st.subheader("📑 Standard RAG (FAISS)")
                    st.info(data["standard_rag_answer"])
                with col_right:
                    st.subheader("🕸️ Microsoft GraphRAG")
                    st.success(data["graph_rag_answer"])