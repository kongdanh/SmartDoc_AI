"""
SmartDoc AI — Streamlit Dashboard
Full-featured UI: Overview, Chat, Query, Knowledge Graph, Upload, Compare RAG
"""

import streamlit as st
import requests
import json
import time
import pandas as pd

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDoc AI — Knowledge Management",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001"

# ─── Premium CSS Theme (matches dashboard.html dark theme) ──────────────────
st.markdown("""
<style>
    /* ───── Import Font ───── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ───── Root Variables ───── */
    :root {
        --bg: #0f1117; --surface: #1a1d27; --surface2: #242836; --border: #2e3348;
        --text: #e4e6f0; --text2: #9498b0; --accent: #6c8aff; --accent-dim: #3d5afe;
        --green: #4caf84; --red: #e05560; --orange: #f0a040; --radius: 10px;
        --gradient-1: linear-gradient(135deg, #6c8aff 0%, #3d5afe 100%);
        --gradient-2: linear-gradient(135deg, #4caf84 0%, #2e7d5a 100%);
        --gradient-3: linear-gradient(135deg, #e05560 0%, #c23040 100%);
    }

    /* ───── Global ───── */
    .stApp { font-family: 'Inter', -apple-system, sans-serif !important; }

    /* ───── Sidebar ───── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13151f 0%, #1a1d27 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.8px;
        color: var(--text2) !important; font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 10px 16px !important; border-radius: 8px !important;
        transition: all 0.2s ease !important; font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(108, 138, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: rgba(108, 138, 255, 0.15) !important; color: var(--accent) !important;
    }

    /* ───── Metric Cards ───── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1d27, #242836) !important;
        padding: 20px 24px !important; border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 32px !important; font-weight: 800 !important;
        background: var(--gradient-1); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; background-clip: text;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: 13px !important; color: var(--text2) !important;
        text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important;
    }

    /* ───── Buttons ───── */
    .stButton > button {
        border-radius: 8px !important; font-weight: 600 !important;
        transition: all 0.2s ease !important; letter-spacing: 0.3px !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }
    .stButton > button[kind="primary"] {
        background: var(--gradient-1) !important; border: none !important;
        box-shadow: 0 4px 15px rgba(61, 90, 254, 0.4) !important;
    }

    /* ───── Expanders ───── */
    .streamlit-expanderHeader {
        background: var(--surface) !important; border-radius: 10px !important;
        border: 1px solid var(--border) !important; font-weight: 600 !important;
    }

    /* ───── Chat Bubbles ───── */
    .chat-bubble-user {
        background: linear-gradient(135deg, #3d5afe, #6c8aff); color: #fff;
        padding: 14px 20px; border-radius: 16px 16px 4px 16px; margin: 8px 0;
        font-size: 14px; line-height: 1.7; max-width: 80%; margin-left: auto;
        box-shadow: 0 4px 15px rgba(61, 90, 254, 0.3);
        animation: msgIn 0.3s ease-out;
    }
    .chat-bubble-assistant {
        background: linear-gradient(145deg, #242836, #1e2130);
        border: 1px solid var(--border); color: var(--text);
        padding: 14px 20px; border-radius: 16px 16px 16px 4px; margin: 8px 0;
        font-size: 14px; line-height: 1.7; max-width: 85%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: msgIn 0.3s ease-out;
    }
    .chat-bubble-assistant strong { color: var(--accent) !important; }
    .chat-bubble-assistant code {
        background: rgba(108, 138, 255, 0.15); padding: 2px 6px;
        border-radius: 4px; font-size: 13px;
    }
    @keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

    /* ───── Compare Cards ───── */
    .compare-card {
        background: linear-gradient(145deg, #1a1d27, #242836);
        border: 1px solid var(--border); border-radius: 14px;
        padding: 24px; margin: 8px 0; min-height: 200px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.2s, border-color 0.2s;
    }
    .compare-card:hover { transform: translateY(-2px); border-color: var(--accent-dim); }
    .compare-card h4 { margin: 0 0 12px; font-size: 16px; font-weight: 700; }
    .compare-card p { color: var(--text2); font-size: 14px; line-height: 1.7; }
    .compare-rag { border-top: 3px solid var(--orange); }
    .compare-graphrag { border-top: 3px solid var(--green); }

    /* ───── File List ───── */
    .file-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 16px; background: var(--surface2); border-radius: 8px;
        margin: 6px 0; border: 1px solid var(--border);
        transition: all 0.2s ease; font-size: 14px;
    }
    .file-item:hover { border-color: var(--accent-dim); background: rgba(108, 138, 255, 0.05); }
    .file-name { font-family: 'Cascadia Code', Consolas, monospace; font-weight: 500; }
    .file-size { color: var(--text2); font-size: 12px; }

    /* ───── Status Chips ───── */
    .chip-ready {
        background: rgba(76, 175, 132, 0.15); color: #4caf84;
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        display: inline-flex; align-items: center; gap: 6px;
    }
    .chip-ready::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #4caf84; }
    .chip-notready {
        background: rgba(224, 85, 96, 0.15); color: #e05560;
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        display: inline-flex; align-items: center; gap: 6px;
    }
    .chip-notready::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #e05560; }

    /* ───── Page Title ───── */
    .page-title {
        font-size: 28px; font-weight: 800; letter-spacing: -0.5px;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #e4e6f0 0%, #9498b0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .page-subtitle { color: var(--text2); font-size: 14px; margin-bottom: 24px; }

    /* ───── Session Card ───── */
    .session-card {
        background: var(--surface2); border: 1px solid var(--border); border-radius: 10px;
        padding: 12px 16px; margin: 4px 0; cursor: pointer;
        transition: all 0.2s; display: flex; justify-content: space-between; align-items: center;
    }
    .session-card:hover { border-color: var(--accent-dim); background: rgba(108, 138, 255, 0.05); }
    .session-title { font-size: 13px; font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-meta { color: var(--text2); font-size: 11px; }

    /* ───── Domain Cards ───── */
    .domain-card {
        background: linear-gradient(145deg, #1a1d27, #242836);
        border: 1px solid var(--border); border-radius: 14px; padding: 20px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }
    .domain-card:hover { transform: translateY(-2px); border-color: var(--accent-dim);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2); }

    /* ───── Info/Success/Error boxes ───── */
    .stAlert > div { border-radius: 10px !important; }

    /* ───── Upload Zone ───── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important; border-radius: 12px !important;
        padding: 8px !important; transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

    /* ───── Tabs ───── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important; font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    /* ───── Table ───── */
    .stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

    /* ───── Divider ───── */
    hr { border-color: var(--border) !important; opacity: 0.5 !important; }

    /* ───── Scrollbar ───── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ───────────────────────────────────────────────────────

def api_get(path):
    """Safe GET request to backend API."""
    try:
        res = requests.get(f"{API_BASE}{path}", timeout=10)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception:
        return None

def api_post(path, json_data=None, files=None):
    """Safe POST request to backend API."""
    try:
        if files:
            res = requests.post(f"{API_BASE}{path}", files=files, timeout=120)
        else:
            res = requests.post(f"{API_BASE}{path}", json=json_data, timeout=120)
        return res
    except Exception as e:
        return None

def api_delete(path):
    """Safe DELETE request to backend API."""
    try:
        res = requests.delete(f"{API_BASE}{path}", timeout=10)
        return res
    except Exception:
        return None

def get_domains():
    """Fetch list of domains from backend."""
    data = api_get("/domains")
    if data and "domains" in data:
        return data["domains"]
    return []

def get_domain_names(only_ready=False):
    """Get domain names, optionally only ready ones."""
    domains = get_domains()
    if only_ready:
        return [d["name"] for d in domains if d.get("ready")]
    return [d["name"] for d in domains]

def format_size(size_bytes):
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1048576:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1048576:.1f} MB"

def check_backend():
    """Check if backend is running."""
    try:
        res = requests.get(f"{API_BASE}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo & Branding
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <div style="font-size: 42px; line-height: 1; margin-bottom: 6px;">🧠</div>
        <div style="font-size: 22px; font-weight: 800; letter-spacing: -0.5px;
             background: linear-gradient(135deg, #6c8aff, #4caf84);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            SmartDoc AI
        </div>
        <div style="font-size: 11px; color: #9498b0; margin-top: 2px; letter-spacing: 0.5px;">
            RAG & GRAPHRAG KNOWLEDGE SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    page = st.radio(
        "NAVIGATION",
        ["📊 Overview", "💬 Chat", "🔍 Query", "🕸️ Knowledge Graph", "📤 Upload Files", "⚖️ Compare RAG"],
        label_visibility="collapsed",
    )

    st.divider()

    # Backend Status
    is_online = check_backend()
    if is_online:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; padding:10px 14px;
             background: rgba(76, 175, 132, 0.1); border-radius:10px;
             border: 1px solid rgba(76, 175, 132, 0.2);">
            <div style="width:8px; height:8px; border-radius:50%; background:#4caf84;
                 box-shadow: 0 0 8px rgba(76,175,132,0.6);"></div>
            <span style="font-size:13px; color:#4caf84; font-weight:600;">Backend Online</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; padding:10px 14px;
             background: rgba(224, 85, 96, 0.1); border-radius:10px;
             border: 1px solid rgba(224, 85, 96, 0.2);">
            <div style="width:8px; height:8px; border-radius:50%; background:#e05560;"></div>
            <span style="font-size:13px; color:#e05560; font-weight:600;">Backend Offline</span>
        </div>""", unsafe_allow_html=True)
        st.caption("Chạy `python main.py` để khởi động server")

    # Chat History in Sidebar
    st.divider()
    st.markdown('<div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:#9498b0; font-weight:600; padding:0 4px; margin-bottom:8px;">💬 Chat History</div>', unsafe_allow_html=True)

    sessions_data = api_get("/chat/sessions")
    if sessions_data and sessions_data.get("sessions"):
        for sess in sessions_data["sessions"][:8]:
            col_s1, col_s2 = st.columns([5, 1])
            with col_s1:
                if st.button(f"📝 {sess['title'][:30]}", key=f"sess_{sess['id']}", use_container_width=True):
                    st.session_state["resume_session"] = sess["id"]
                    st.session_state["force_page"] = "💬 Chat"
                    st.rerun()
            with col_s2:
                if st.button("🗑️", key=f"del_sess_{sess['id']}"):
                    api_delete(f"/chat/sessions/{sess['id']}")
                    st.rerun()

        if st.button("🗑️ Xóa toàn bộ lịch sử", use_container_width=True, type="secondary"):
            for sess in sessions_data["sessions"]:
                api_delete(f"/chat/sessions/{sess['id']}")
            st.rerun()
    else:
        st.caption("Chưa có lịch sử chat")

    # Tech Stack
    st.divider()
    st.markdown("""
    <div style="font-size: 10px; color: #9498b0; text-align:center; line-height: 1.6;">
        <div style="font-weight:600; margin-bottom:4px;">🛠 Tech Stack</div>
        FastAPI · GraphRAG · ChromaDB<br>
        LangChain · Streamlit · D3.js<br>
        OpenRouter LLM · HuggingFace
    </div>
    """, unsafe_allow_html=True)


# ─── Handle page override from sidebar session click ────────────────────────
if "force_page" in st.session_state:
    page = st.session_state.pop("force_page")


# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="page-title">📊 System Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Monitoring your SmartDoc AI knowledge domains</div>', unsafe_allow_html=True)

    domains_data = get_domains()

    # Stats Row
    col1, col2, col3, col4 = st.columns(4)
    total = len(domains_data)
    ready = len([d for d in domains_data if d.get("ready")])
    files = sum(d.get("indexed_files", 0) for d in domains_data)
    col1.metric("📁 Total Domains", total)
    col2.metric("✅ Ready", ready)
    col3.metric("📄 Indexed Files", files)
    col4.metric("⚡ System", "Online" if is_online else "Offline")

    st.divider()

    # Action Bar
    col_act1, col_act2 = st.columns([3, 1])
    with col_act1:
        st.markdown("### 📂 Domain Management")
    with col_act2:
        if st.button("🔄 Re-index All", type="primary", use_container_width=True):
            res = api_post("/reindex")
            if res and res.status_code == 200:
                st.toast("🔄 Re-indexing tất cả domains...", icon="✅")
            else:
                st.toast("❌ Re-index thất bại", icon="❌")

    if not domains_data:
        st.info("📁 Chưa có domain nào. Hãy vào **Upload Files** để tải tài liệu lên.")
    else:
        # Domain Cards Grid
        cols = st.columns(min(3, len(domains_data)))
        for i, d in enumerate(domains_data):
            with cols[i % 3]:
                status_chip = "chip-ready" if d.get("ready") else "chip-notready"
                status_text = "Ready" if d.get("ready") else "Not Ready"
                st.markdown(f"""
                <div class="domain-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-size:18px; font-weight:700;">📁 {d['name']}</span>
                        <span class="{status_chip}">{status_text}</span>
                    </div>
                    <div style="color:#9498b0; font-size:13px; margin-bottom:12px;">
                        📄 {d.get('indexed_files', 0)} indexed file(s)
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("🔄 Re-index", key=f"ri_{d['name']}", use_container_width=True):
                        api_post(f"/reindex/{d['name']}")
                        st.toast(f"🔄 Re-indexing {d['name']}...")
                with col_b:
                    if st.button("📄 Files", key=f"fv_{d['name']}", use_container_width=True):
                        st.session_state["view_domain"] = d["name"]
                with col_c:
                    if st.button("🗑️ Delete", key=f"dd_{d['name']}", use_container_width=True):
                        api_delete(f"/domains/{d['name']}")
                        st.rerun()

        # File Viewer for selected domain
        if "view_domain" in st.session_state:
            domain_name = st.session_state["view_domain"]
            st.divider()
            st.markdown(f"### 📄 Files in `{domain_name}`")

            files_data = api_get(f"/domains/{domain_name}/files")
            if files_data and files_data.get("files"):
                for f in files_data["files"]:
                    col_f1, col_f2, col_f3 = st.columns([5, 2, 1])
                    with col_f1:
                        st.markdown(f'<span class="file-name">📄 {f["name"]}</span>', unsafe_allow_html=True)
                    with col_f2:
                        st.caption(format_size(f["size"]))
                    with col_f3:
                        if st.button("🗑️", key=f"dfl_{domain_name}_{f['name']}"):
                            api_delete(f"/domains/{domain_name}/files/{f['name']}")
                            st.rerun()
            else:
                st.caption("Không có file nào trong domain này.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHAT PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat":
    st.markdown('<div class="page-title">💬 AI Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Trò chuyện với AI được tăng cường bởi Knowledge Base (GraphRAG)</div>', unsafe_allow_html=True)

    # Chat controls
    col_ch1, col_ch2, col_ch3 = st.columns([3, 1, 1])
    with col_ch1:
        domain_names = get_domain_names(only_ready=True)
        all_domain_names = get_domain_names()
        chat_domains = domain_names if domain_names else all_domain_names
        if chat_domains:
            domain = st.selectbox("🌐 Domain", chat_domains, key="chat_domain", label_visibility="collapsed")
        else:
            domain = None
            st.warning("Không có domain. Hãy upload tài liệu trước.")
    with col_ch2:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.pop("chat_session_id", None)
            st.session_state["messages"] = []
            st.rerun()
    with col_ch3:
        session_info = st.session_state.get("chat_session_id", "—")
        st.caption(f"Session: `{session_info}`")

    st.divider()

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Handle session resume
    if "resume_session" in st.session_state:
        sess_id = st.session_state.pop("resume_session")
        sess_data = api_get(f"/chat/sessions/{sess_id}")
        if sess_data:
            st.session_state["chat_session_id"] = sess_data["id"]
            st.session_state["messages"] = [
                {"role": m["role"], "content": m["content"]} for m in sess_data.get("messages", [])
            ]

    # Display messages
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
             padding: 60px 20px; color: #9498b0; text-align:center;">
            <div style="font-size:56px; margin-bottom:16px;">🧠</div>
            <div style="font-size:20px; font-weight:700; color:#e4e6f0; margin-bottom:8px;">SmartDoc AI Chat</div>
            <div style="font-size:14px; max-width:400px; line-height:1.6;">
                Chọn domain và đặt câu hỏi — Câu trả lời được tăng cường bởi Knowledge Graph của bạn.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat Input
    if domain:
        if prompt := st.chat_input("💡 Hỏi bất cứ điều gì về knowledge base...", key="chat_input"):
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

            # Stream response
            with st.spinner("🧠 AI đang suy nghĩ..."):
                full_response = ""
                response_placeholder = st.empty()

                try:
                    session_id = st.session_state.get("chat_session_id")
                    payload = {"domain": domain, "message": prompt}
                    if session_id:
                        payload["session_id"] = session_id

                    res = requests.post(f"{API_BASE}/chat", json=payload, stream=True, timeout=120)

                    if res.status_code == 200:
                        for line in res.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    try:
                                        data = json.loads(decoded[6:])
                                        if data.get("type") == "start":
                                            st.session_state["chat_session_id"] = data.get("session_id")
                                        elif data.get("type") == "token":
                                            full_response += data.get("content", "")
                                            response_placeholder.markdown(
                                                f'<div class="chat-bubble-assistant">{full_response}▌</div>',
                                                unsafe_allow_html=True
                                            )
                                    except json.JSONDecodeError:
                                        pass

                        response_placeholder.markdown(
                            f'<div class="chat-bubble-assistant">{full_response}</div>',
                            unsafe_allow_html=True
                        )
                        st.session_state["messages"].append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"❌ API Error: {res.status_code}")

                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUERY PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Query":
    st.markdown('<div class="page-title">🔍 Advanced Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Truy vấn sâu vào Knowledge Graph với nhiều phương pháp tìm kiếm</div>', unsafe_allow_html=True)

    domain_names = get_domain_names(only_ready=True)

    col_q1, col_q2, col_q3 = st.columns([3, 2, 1])
    with col_q1:
        if domain_names:
            q_domain = st.selectbox("🌐 Domain", domain_names, key="q_domain")
        else:
            q_domain = None
            st.warning("Không có domain ready.")
    with col_q2:
        q_method = st.selectbox("🔧 Method", ["local", "global", "drift", "search"], key="q_method",
                                help="**local**: Entity-focused | **global**: Dataset-wide | **drift**: Hybrid | **search**: Direct text")
    with col_q3:
        q_level = st.slider("📊 Level", 1, 5, 2, key="q_level", help="Community hierarchy level")

    query_text = st.text_area("📝 Query", placeholder="Nhập câu hỏi truy vấn sâu vào knowledge base...", height=100, key="q_text")

    if st.button("🚀 Execute Search", type="primary", disabled=not q_domain, use_container_width=True):
        if not query_text.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            t0 = time.time()
            with st.spinner("🔍 Đang tìm kiếm..."):
                endpoint = f"/query/{q_method}"
                if q_method == "search":
                    body = {"domain": q_domain, "query": query_text.strip()}
                else:
                    body = {
                        "domain": q_domain, "query": query_text.strip(),
                        "community_level": q_level, "response_type": "Multiple Paragraphs"
                    }

                res = api_post(endpoint, json_data=body)

            elapsed = time.time() - t0

            if res and res.status_code == 200:
                data = res.json()
                st.success(f"✅ Hoàn thành trong {elapsed:.1f}s | Method: `{data.get('method', q_method)}`")

                if q_method == "search":
                    # Direct search results
                    tab1, tab2, tab3 = st.tabs(["🏷️ Entities", "🔗 Relationships", "📚 Sources"])
                    with tab1:
                        if data.get("entities"):
                            for e in data["entities"]:
                                with st.expander(f"**{e.get('title', 'N/A')}** `{e.get('type', '')}`"):
                                    st.write(e.get("description", "No description"))
                        else:
                            st.caption("Không tìm thấy entities.")
                    with tab2:
                        if data.get("relationships"):
                            for r in data["relationships"]:
                                st.markdown(f"**{r.get('source', '')}** → **{r.get('target', '')}** *(weight: {r.get('weight', 0)})*")
                                st.caption(r.get("description", ""))
                        else:
                            st.caption("Không tìm thấy relationships.")
                    with tab3:
                        if data.get("sources"):
                            for s in data["sources"]:
                                title = s.get("title", "")
                                body_text = s.get("summary", "") or s.get("text", "")
                                if title:
                                    st.markdown(f"**{title}**")
                                st.caption(body_text[:500])
                                st.divider()
                        else:
                            st.caption("Không tìm thấy sources.")
                else:
                    # LLM response
                    st.markdown("### 📋 Result")
                    st.markdown(data.get("response", "Không có kết quả."))
            else:
                st.error("❌ Query failed. Kiểm tra domain đã được index chưa.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. KNOWLEDGE GRAPH PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🕸️ Knowledge Graph":
    st.markdown('<div class="page-title">🕸️ Visual Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Trực quan hóa mạng lưới tri thức được trích xuất bởi GraphRAG</div>', unsafe_allow_html=True)

    domain_names = get_domain_names(only_ready=True)

    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        if domain_names:
            g_domain = st.selectbox("🌐 Domain", domain_names, key="g_domain")
        else:
            g_domain = None
            st.warning("Không có domain ready.")
    with col_g2:
        load_btn = st.button("🚀 Load Graph", type="primary", disabled=not g_domain, use_container_width=True)

    if load_btn and g_domain:
        with st.spinner("🕸️ Đang tải Knowledge Graph..."):
            data = api_get(f"/graph/{g_domain}")

        if data and not data.get("error"):
            # Stats
            col_gs1, col_gs2, col_gs3 = st.columns(3)
            col_gs1.metric("🔵 Nodes", len(data.get("nodes", [])))
            col_gs2.metric("🔗 Edges", len(data.get("edges", [])))
            col_gs3.metric("📝 Text Units", data.get("text_units", 0))

            st.divider()

            # Render with streamlit-agraph
            try:
                from streamlit_agraph import agraph, Node, Edge, Config

                max_deg = max((n.get("degree", 1) for n in data["nodes"]), default=1)

                nodes = []
                for n in data["nodes"]:
                    ratio = n.get("degree", 1) / max_deg if max_deg > 0 else 0
                    if ratio > 0.5:
                        color = "#e05560"
                    elif ratio > 0.2:
                        color = "#6c8aff"
                    else:
                        color = "#4caf84"

                    label = n["id"][:20] + "..." if len(n["id"]) > 20 else n["id"]
                    nodes.append(Node(
                        id=n["id"], label=label,
                        size=12 + (n.get("degree", 1) / max_deg) * 28,
                        color=color, font={"color": "#e4e6f0", "size": 11},
                        title=f"{n['id']}\nType: {n.get('type', '')}\nDegree: {n.get('degree', 0)}\n{n.get('description', '')[:200]}"
                    ))

                edges = []
                for e in data["edges"]:
                    edges.append(Edge(
                        source=e["source"], target=e["target"],
                        color="#3e4460", width=max(0.5, min(e.get("weight", 1) / 5, 3)),
                        title=e.get("description", "")[:100]
                    ))

                config = Config(
                    width=1200, height=650, directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#6c8aff",
                    collapsible=True,
                    node={"labelProperty": "label"},
                    link={"labelProperty": "label", "renderLabel": False},
                    staticGraphWithDragAndDrop=False,
                    physics=True,
                )

                agraph(nodes=nodes, edges=edges, config=config)

            except ImportError:
                st.warning("⚠️ Cần cài `streamlit-agraph`: `pip install streamlit-agraph`")

            # Entity Table
            st.divider()
            with st.expander("📋 Entity Details Table", expanded=False):
                if data.get("nodes"):
                    df = pd.DataFrame(data["nodes"])
                    display_cols = [c for c in ["id", "type", "degree", "description"] if c in df.columns]
                    df = df[display_cols].rename(columns={
                        "id": "Entity", "type": "Type", "degree": "Degree", "description": "Description"
                    })
                    st.dataframe(df, use_container_width=True, height=400)

        else:
            st.error("❌ Không thể tải graph data. Kiểm tra domain đã index chưa.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. UPLOAD FILES PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Files":
    st.markdown('<div class="page-title">📤 Data Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload tài liệu để xây dựng Knowledge Graph & RAG Index</div>', unsafe_allow_html=True)

    # Upload Section
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        new_domain = st.text_input("🏷️ Domain Name", placeholder="Ví dụ: tai_lieu_cntt", key="u_domain",
                                   help="Tên thư mục chứa tài liệu. Nếu domain đã tồn tại, file sẽ được thêm vào.")
    with col_u2:
        auto_reindex = st.checkbox("🔄 Auto Re-index GraphRAG", value=True, key="u_reindex",
                                   help="Tự động chạy GraphRAG indexing sau khi upload")

    uploaded_files = st.file_uploader(
        "📎 Chọn files để upload (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "doc"],
        help="Hỗ trợ: PDF, DOCX/DOC, TXT — File sẽ được lưu vào resources/uploads/{loại file}"
    )

    if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
        if not new_domain:
            st.warning("⚠️ Nhập tên domain trước.")
        elif not uploaded_files:
            st.warning("⚠️ Chọn ít nhất một file.")
        else:
            with st.spinner("📤 Đang upload và xử lý..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded_files]
                res = api_post(f"/domains/{new_domain}/upload", files=files_payload)

            if res and res.status_code == 200:
                data = res.json()
                st.success(f"✅ Đã upload {data.get('uploaded', 0)} file vào domain `{new_domain}`")
                st.info(data.get("message", ""))

                if auto_reindex:
                    with st.spinner("🔄 Đang chạy GraphRAG indexing..."):
                        api_post(f"/reindex/{new_domain}")
                    st.toast("🔄 GraphRAG indexing đã bắt đầu!")
            else:
                st.error("❌ Upload thất bại. Kiểm tra backend.")

    st.divider()

    # File type info
    st.markdown("""
    <div style="display:flex; gap:12px; margin-bottom:20px;">
        <div style="flex:1; background: linear-gradient(145deg, #1a1d27, #242836); border:1px solid #2e3348; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; margin-bottom:4px;">📕</div>
            <div style="font-size:13px; font-weight:600; color:#e4e6f0;">PDF</div>
            <div style="font-size:11px; color:#9498b0;">Tài liệu, báo cáo</div>
        </div>
        <div style="flex:1; background: linear-gradient(145deg, #1a1d27, #242836); border:1px solid #2e3348; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; margin-bottom:4px;">📘</div>
            <div style="font-size:13px; font-weight:600; color:#e4e6f0;">DOCX</div>
            <div style="font-size:11px; color:#9498b0;">Word documents</div>
        </div>
        <div style="flex:1; background: linear-gradient(145deg, #1a1d27, #242836); border:1px solid #2e3348; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; margin-bottom:4px;">📝</div>
            <div style="font-size:13px; font-weight:600; color:#e4e6f0;">TXT</div>
            <div style="font-size:11px; color:#9498b0;">Plain text</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Existing Files Browser
    st.markdown("### 📂 Quản lý Files theo Domain")

    all_domain_names = get_domain_names()
    if all_domain_names:
        browse_domain = st.selectbox("Chọn domain để xem files:", all_domain_names, key="browse_domain")

        if browse_domain:
            files_data = api_get(f"/domains/{browse_domain}/files")
            if files_data and files_data.get("files"):
                st.markdown(f"**{len(files_data['files'])} file(s)** in `{browse_domain}`")

                for f in files_data["files"]:
                    col_bf1, col_bf2, col_bf3 = st.columns([5, 2, 1])
                    with col_bf1:
                        icon = "📄"
                        ext = f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
                        if ext == "pdf":
                            icon = "📕"
                        elif ext in ("doc", "docx"):
                            icon = "📘"
                        elif ext == "txt":
                            icon = "📝"
                        st.markdown(f"{icon} `{f['name']}`")
                    with col_bf2:
                        st.caption(format_size(f["size"]))
                    with col_bf3:
                        if st.button("🗑️", key=f"del_f_{browse_domain}_{f['name']}"):
                            res = api_delete(f"/domains/{browse_domain}/files/{f['name']}")
                            if res and res.status_code == 200:
                                st.toast(f"✅ Đã xóa {f['name']}")
                                st.rerun()
                            else:
                                st.error(f"❌ Không thể xóa {f['name']}")

                st.divider()
                if st.button(f"🗑️ Xóa toàn bộ domain `{browse_domain}`", type="secondary"):
                    res = api_delete(f"/domains/{browse_domain}")
                    if res and res.status_code == 200:
                        st.toast(f"✅ Đã xóa domain {browse_domain}")
                        st.rerun()
            else:
                st.info(f"📁 Domain `{browse_domain}` chưa có file nào.")
    else:
        st.info("📁 Chưa có domain nào. Upload files ở trên để bắt đầu.")


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARE RAG PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare RAG":
    st.markdown('<div class="page-title">⚖️ Compare RAG vs GraphRAG</div>', unsafe_allow_html=True)

    all_domain_names = get_domain_names()

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        if all_domain_names:
            c_domain = st.selectbox("🌐 Domain", all_domain_names, key="comp_domain")
        else:
            c_domain = None
            st.warning("Không có domain. Upload tài liệu trước.")
    with col_c2:
        if st.button("➕ New Comparison", use_container_width=True, type="primary"):
            st.session_state["compare_messages"] = []
            st.rerun()

    st.divider()

    # Headers for the two columns
    col_hdr1, col_hdr2 = st.columns(2)
    with col_hdr1:
        st.markdown("<h4 style='text-align: center; color: var(--orange); margin-bottom: 16px;'>📑 Standard RAG</h4>", unsafe_allow_html=True)
    with col_hdr2:
        st.markdown("<h4 style='text-align: center; color: var(--green); margin-bottom: 16px;'>🕸️ GraphRAG</h4>", unsafe_allow_html=True)

    if "compare_messages" not in st.session_state:
        st.session_state["compare_messages"] = []

    # Display chat history
    for msg in st.session_state["compare_messages"]:
        col_msg1, col_msg2 = st.columns(2)
        if msg["role"] == "user":
            with col_msg1:
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            with col_msg2:
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with col_msg1:
                st.markdown(f'<div class="chat-bubble-assistant">{msg["standard_rag"]}</div>', unsafe_allow_html=True)
            with col_msg2:
                st.markdown(f'<div class="chat-bubble-assistant">{msg["graph_rag"]}</div>', unsafe_allow_html=True)

    # Chat input
    if c_domain:
        if prompt := st.chat_input("💡 Nhập câu hỏi so sánh...", key="comp_chat_input"):
            # Immediate display
            col_msg1, col_msg2 = st.columns(2)
            with col_msg1:
                st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)
            with col_msg2:
                st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

            st.session_state["compare_messages"].append({"role": "user", "content": prompt})

            with st.spinner("⚖️ Đang so sánh hai hệ thống..."):
                res = api_post("/api/compare-rag", json_data={"query": prompt.strip(), "domain": c_domain})

                if res and res.status_code == 200:
                    data = res.json()
                    std_ans = data.get("standard_rag_answer", "Không có câu trả lời.")
                    graph_ans = data.get("graph_rag_answer", "Không có câu trả lời.")
                    
                    st.session_state["compare_messages"].append({
                        "role": "assistant",
                        "standard_rag": std_ans,
                        "graph_rag": graph_ans
                    })
                    
                    col_ans1, col_ans2 = st.columns(2)
                    with col_ans1:
                        st.markdown(f'<div class="chat-bubble-assistant">{std_ans}</div>', unsafe_allow_html=True)
                    with col_ans2:
                        st.markdown(f'<div class="chat-bubble-assistant">{graph_ans}</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ So sánh thất bại. Kiểm tra backend.")
