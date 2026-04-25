import streamlit as st
import pandas as pd
from ui.api import get_domain_names, api_get

@st.cache_data(ttl=300) # Lưu bộ đệm 5 phút để tránh lag máy
def fetch_graph_data(domain):
    return api_get(f"/graph/{domain}")

def render_graph():
    st.header("Knowledge Graph Visualizer")
    st.caption("Trực quan hóa mạng lưới thực thể và quan hệ được trích xuất bởi GraphRAG")
    st.divider()

    domain_names = get_domain_names(only_ready=True)
    
    col1, col2 = st.columns([4, 1])
    g_domain = col1.selectbox("Chọn Domain", domain_names if domain_names else ["Chưa có dữ liệu"], label_visibility="collapsed")
    load_btn = col2.button("Hiển thị Đồ thị", type="primary", use_container_width=True)

    if load_btn and g_domain and g_domain != "Chưa có dữ liệu":
        data = fetch_graph_data(g_domain)

        if data and not data.get("error"):
            # Các chỉ số thống kê
            m1, m2, m3 = st.columns(3)
            m1.metric("Thực thể (Nodes)", len(data.get("nodes", [])))
            m2.metric("Quan hệ (Edges)", len(data.get("edges", [])))
            m3.metric("Số lượng Chunk", data.get("text_units", 0))

            st.write("") # Dòng trống

            # Vùng render đồ thị Agraph
            with st.container(border=True):
                try:
                    from streamlit_agraph import agraph, Node, Edge, Config

                    max_deg = max((n.get("degree", 1) for n in data["nodes"]), default=1)
                    nodes = []
                    for n in data["nodes"]:
                        ratio = n.get("degree", 1) / max_deg if max_deg > 0 else 0
                        color = "#e05560" if ratio > 0.5 else ("#6c8aff" if ratio > 0.2 else "#4caf84")
                        label = n["id"][:20] + "..." if len(n["id"]) > 20 else n["id"]
                        
                        nodes.append(Node(
                            id=n["id"], label=label,
                            size=12 + ratio * 28,
                            color=color, font={"color": "#e4e6f0", "size": 11},
                            title=f"{n['id']}\nType: {n.get('type', '')}\nDegree: {n.get('degree', 0)}"
                        ))

                    edges = [Edge(source=e["source"], target=e["target"], color="#3e4460", width=max(0.5, min(e.get("weight", 1) / 5, 3))) for e in data["edges"]]

                    config = Config(
                        width="100%", height=650, directed=True,
                        nodeHighlightBehavior=True, highlightColor="#6c8aff",
                        collapsible=True, node={"labelProperty": "label"},
                        link={"labelProperty": "label", "renderLabel": False},
                        staticGraphWithDragAndDrop=False, physics=True,
                    )

                    st.caption("*Kéo thả chuột để di chuyển, cuộn chuột để phóng to/thu nhỏ.*")
                    agraph(nodes=nodes, edges=edges, config=config)
                except ImportError:
                    st.warning("Cần cài thư viện: `pip install streamlit-agraph`")

            # Bảng chi tiết thực thể
            with st.expander("Bảng chi tiết thực thể (Entity Details)", expanded=False):
                if data.get("nodes"):
                    df = pd.DataFrame(data["nodes"])
                    display_cols = [c for c in ["id", "type", "degree", "description"] if c in df.columns]
                    df = df[display_cols]
                    st.dataframe(df, use_container_width=True, height=400)
        else:
            st.error("Không thể tải dữ liệu. Domain này có thể chưa được Index thành công.")