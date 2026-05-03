import streamlit as st

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

from ui.sidebar import render_sidebar
from ui.pages.overview import render_overview
from ui.pages.chat import render_chat
from ui.pages.query import render_query
from ui.pages.graph import render_graph
from ui.pages.compare import render_compare

render_sidebar()

pages = ["Overview", "Chat", "Query", "Knowledge Graph", "Compare RAG"]

nav_cols = st.columns(len(pages))

for i, p in enumerate(pages):
    with nav_cols[i]:
        b_type = "primary" if st.session_state["page"] == p else "secondary"
        if st.button(p, key=f"nav_{p}", use_container_width=True, type=b_type):
            st.session_state["page"] = p
            st.rerun()

st.divider()

p = st.session_state["page"]
if p == "Overview":
    render_overview()
elif p == "Chat":
    render_chat()
elif p == "Query":
    render_query()
elif p == "Knowledge Graph":
    render_graph()
elif p == "Compare RAG":
    render_compare()