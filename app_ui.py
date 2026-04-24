import streamlit as st

# 1. Khởi tạo cấu hình trang (Chỉ dùng Native)
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Khởi tạo trạng thái trang
if "page" not in st.session_state:
    st.session_state["page"] = "Chat"

# 3. Import các trang (Đảm bảo bạn đã dùng bộ code các trang tôi gửi ở lần trước)
from ui.sidebar import render_sidebar
from ui.pages.overview import render_overview
from ui.pages.chat import render_chat
from ui.pages.query import render_query
from ui.pages.graph import render_graph
from ui.pages.compare import render_compare

# 4. Render Sidebar
render_sidebar()

# 5. Thanh điều hướng Topbar (Dùng cột Native, nói KHÔNG với CSS)
pages = ["Overview", "Chat", "Query", "Knowledge Graph", "Compare RAG"]

# Chia cột đều nhau cho menu
nav_cols = st.columns(len(pages))

for i, p in enumerate(pages):
    with nav_cols[i]:
        # Trang nào đang chọn thì nút có màu Primary (xanh), còn lại là Secondary (xám)
        b_type = "primary" if st.session_state["page"] == p else "secondary"
        if st.button(p, key=f"nav_{p}", use_container_width=True, type=b_type):
            st.session_state["page"] = p
            st.rerun()

st.divider() # Đường kẻ ngang phân tách Menu và Nội dung (Tạo không gian thở)

# 6. Render Nội dung trang
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