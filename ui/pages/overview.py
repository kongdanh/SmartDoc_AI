import streamlit as st
from ui.api import get_domains, api_post, api_delete

def render_overview():
    st.title("Data Management")
    st.write("Manage documents and knowledge graphs.")
    st.divider()

    domains_data = get_domains()
    
    st.subheader("System Statistics")
    col1, col2, col3 = st.columns(3)
    total = len(domains_data)
    ready = len([d for d in domains_data if d.get("ready")])
    files = sum(d.get("indexed_files", 0) for d in domains_data)
    
    col1.metric("Total Domains", total)
    col2.metric("Ready Domains", ready)
    col3.metric("Indexed Files", files)

    st.divider()

    st.subheader("Upload Documents")
    new_domain = st.text_input("Enter domain name (no spaces):")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    auto_reindex = st.checkbox("Auto-reindex after upload", value=True)

    if st.button("Process Data", type="primary"):
        if not new_domain or not uploaded_files:
            st.warning("Please enter domain name and select files.")
        else:
            with st.spinner("Uploading..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                res = api_post(f"/domains/{new_domain}/upload", files=files_payload)
                
                if res and res.status_code == 200:
                    if auto_reindex:
                        api_post(f"/reindex/{new_domain}")
                        st.success("Upload successful! Indexing in progress.")
                    else:
                        st.success("Upload successful!")
                else:
                    st.error("Upload failed.")

    st.divider()

    st.subheader("Domain List")
    if not domains_data:
        st.info("No domains.")
    else:
        for d in domains_data:
            with st.expander(f"{d['name']} ({'Ready' if d.get('ready') else 'Processing'})"):
                st.write(f"Files indexed: **{d.get('indexed_files', 0)}**")
                c1, c2 = st.columns(2)
                if c1.button("Update Index", key=f"re_{d['name']}", use_container_width=True):
                    api_post(f"/reindex/{d['name']}")
                    st.toast("Update command sent!")
                if c2.button("Delete", key=f"del_{d['name']}", use_container_width=True):
                    api_delete(f"/domains/{d['name']}")
                    st.rerun()