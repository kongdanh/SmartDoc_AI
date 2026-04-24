import requests
import json
import streamlit as st

API_BASE = "http://localhost:8001"

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
    except Exception:
        return None

def api_delete(path):
    """Safe DELETE request to backend API."""
    try:
        res = requests.delete(f"{API_BASE}{path}", timeout=10)
        return res
    except Exception:
        return None

# 🔴 THÊM CACHE: Lưu dữ liệu trong 10 giây để chuyển trang tức thì
@st.cache_data(ttl=10)
def get_domains():
    """Fetch list of domains from backend."""
    data = api_get("/domains")
    if data and "domains" in data:
        return data["domains"]
    return []

@st.cache_data(ttl=10)
def get_domain_names(only_ready=False):
    """Get domain names, optionally only ready ones."""
    domains = get_domains()
    if only_ready:
        return [d["name"] for d in domains if d.get("ready")]
    return [d["name"] for d in domains]

# 🔴 THÊM HÀM CACHE LỊCH SỬ CHAT
@st.cache_data(ttl=10)
def get_chat_sessions_cached():
    return api_get("/chat/sessions")

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