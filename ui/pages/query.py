import streamlit as st
import time
from ui.api import get_domain_names, api_post

def render_query():
    st.title("Truy Vấn Nâng Cao")
    st.write("Sử dụng các phương pháp tìm kiếm chuyên sâu vào Graph Database.")

    domain_names = get_domain_names(only_ready=True)
    
    # Bảng điều khiển chọn thông số
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
        q_domain = c1.selectbox("Nguồn dữ liệu", domain_names if domain_names else ["Chưa có dữ liệu"])
        q_method = c2.selectbox("Phương pháp", ["local", "global", "drift", "search"])
        q_level = c3.select_slider("Cấp độ (Level)", options=[1, 2, 3, 4, 5], value=2)

    query_text = st.chat_input("Nhập câu lệnh truy vấn của bạn...")

    if query_text:
        if not q_domain or q_domain == "Chưa có dữ liệu":
            st.error("Vui lòng chọn Domain khả dụng.")
            return

        t0 = time.time()
        with st.status(f"Đang chạy phương pháp `{q_method}`...", expanded=True) as status:
            st.write("Đang kết nối tới GraphRAG Engine...")
            endpoint = f"/query/{q_method}"
            
            body = {
                "domain": q_domain, "query": query_text.strip(),
                "community_level": q_level, "response_type": "Multiple Paragraphs"
            } if q_method != "search" else {"domain": q_domain, "query": query_text.strip()}
            
            res = api_post(endpoint, json_data=body)
            elapsed = time.time() - t0

            if res and res.status_code == 200:
                data = res.json()
                status.update(label=f"Truy vấn xong trong {elapsed:.1f}s", state="complete")
                
                st.subheader("Kết quả phân tích")
                if q_method == "search":
                    t1, t2, t3 = st.tabs(["Thực thể", "Quan hệ", "Nguồn trích dẫn"])
                    with t1: st.json(data.get("entities", []))
                    with t2: st.json(data.get("relationships", []))
                    with t3: st.json(data.get("sources", []))
                else:
                    st.info(data.get("response", "Hệ thống không tìm thấy kết quả phù hợp."))
            else:
                status.update(label="Lỗi truy vấn", state="error")
                detail = f" ({res.status_code})" if res else ""
                st.error(f"Lỗi từ server Backend{detail}.")