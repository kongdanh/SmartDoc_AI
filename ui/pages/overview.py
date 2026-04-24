import streamlit as st
from ui.api import get_domains, api_post, api_delete

def render_overview():
    st.title("📊 Quản Lý Dữ Liệu (Overview)")
    st.write("Kiểm soát tài liệu và không gian tri thức GraphRAG của bạn.")
    st.divider()

    domains_data = get_domains()
    
    # Khu vực Thống kê
    st.subheader("📈 Thống kê hệ thống")
    col1, col2, col3 = st.columns(3)
    total = len(domains_data)
    ready = len([d for d in domains_data if d.get("ready")])
    files = sum(d.get("indexed_files", 0) for d in domains_data)
    
    col1.metric("Số lượng Domain", total)
    col2.metric("Domain Sẵn sàng", ready)
    col3.metric("Tài liệu đã Index", files)

    st.divider()

    # Khu vực Upload
    st.subheader("📤 Tải lên tài liệu mới")
    new_domain = st.text_input("Nhập tên Domain (viết liền không dấu):")
    uploaded_files = st.file_uploader("Kéo thả file tài liệu vào đây", accept_multiple_files=True)
    auto_reindex = st.checkbox("Tự động Index GraphRAG sau khi tải lên", value=True)

    if st.button("Bắt đầu xử lý dữ liệu", type="primary"):
        if not new_domain or not uploaded_files:
            st.warning("Vui lòng nhập tên Domain và chọn file.")
        else:
            with st.spinner("Đang tải dữ liệu lên server..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                res = api_post(f"/domains/{new_domain}/upload", files=files_payload)
                
                if res and res.status_code == 200:
                    if auto_reindex:
                        api_post(f"/reindex/{new_domain}")
                        st.success("Tải lên thành công! Hệ thống đang Index ngầm.")
                    else:
                        st.success("Tải lên thành công!")
                else:
                    st.error("Lỗi tải lên.")

    st.divider()

    # Quản lý Domain
    st.subheader("📂 Danh sách Domain hiện tại")
    if not domains_data:
        st.info("Trống.")
    else:
        for d in domains_data:
            with st.expander(f"📁 {d['name']} ({'Sẵn sàng' if d.get('ready') else 'Đang xử lý'})"):
                st.write(f"Số tệp bên trong: **{d.get('indexed_files', 0)}**")
                c1, c2 = st.columns(2)
                if c1.button("🔄 Cập nhật Index", key=f"re_{d['name']}", use_container_width=True):
                    api_post(f"/reindex/{d['name']}")
                    st.toast("Đã gửi lệnh cập nhật!")
                if c2.button("🗑️ Xóa Domain", key=f"del_{d['name']}", use_container_width=True):
                    api_delete(f"/domains/{d['name']}")
                    st.rerun()