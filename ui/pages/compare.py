import streamlit as st
from ui.api import get_domain_names, api_post

def render_compare():
    st.markdown("<h2>So Sanh Standard RAG vs GraphRAG</h2>", unsafe_allow_html=True)
    
    all_domain_names = get_domain_names()

    # Domain selector
    col1, col2 = st.columns([4, 1])
    selected_domain = col1.selectbox(
        "Chon Domain", 
        all_domain_names if all_domain_names else ["Chua co du lieu"], 
        label_visibility="collapsed",
        key="compare_domain"
    )
    if col2.button("Lam moi", use_container_width=True, key="clear_compare"):
        st.session_state["compare_qa_pairs"] = []
        st.rerun()

    st.divider()
    
    # Initialize data
    if "compare_qa_pairs" not in st.session_state:
        st.session_state["compare_qa_pairs"] = []  # Each: {"question": str, "standard": str, "graph": str}
    
    if not selected_domain or selected_domain == "Chua co du lieu":
        st.warning("Vui long chon Domain co du lieu")
        return
    
    # Title row for two columns
    title_col1, title_col2 = st.columns(2, gap="large")
    with title_col1:
        st.markdown("### Standard RAG")
        st.caption("Vector Search + Semantic Similarity")
    with title_col2:
        st.markdown("### GraphRAG")
        st.caption("Knowledge Graph + Community Detection")
    
    st.divider()
    
    # SINGLE CONTAINER for both columns - only ONE scrollbar
    main_container = st.container(height=450, border=False)
    
    with main_container:
        # Use HTML/CSS table-like layout to ensure alignment
        st.markdown("""
        <style>
        .compare-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .compare-col {
            flex: 1;
            overflow-wrap: break-word;
            word-wrap: break-word;
        }
        .question-badge {
            background-color: #2d2d3d;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .answer-bubble {
            background-color: #1e1e2a;
            padding: 12px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if not st.session_state["compare_qa_pairs"]:
            st.info("Chua co cau hoi. Hay nhap cau hoi o phia duoi.")
        else:
            # Display each Q&A pair in a synchronized row
            for idx, pair in enumerate(st.session_state["compare_qa_pairs"]):
                # Create two columns for this pair
                c1, c2 = st.columns(2, gap="large")
                
                # Question (same for both columns)
                with c1:
                    st.markdown(f"**Cau hoi {idx + 1}:** {pair['question']}")
                with c2:
                    st.markdown(f"**Cau hoi {idx + 1}:** {pair['question']}")
                
                # Standard RAG answer in left column
                with c1:
                    with st.chat_message("assistant"):
                        st.markdown(pair["standard"])
                
                # GraphRAG answer in right column
                with c2:
                    with st.chat_message("assistant"):
                        st.markdown(pair["graph"])
                
                # Separator between pairs
                if idx < len(st.session_state["compare_qa_pairs"]) - 1:
                    st.divider()
    
    st.divider()
    
    # Chat input at bottom
    if prompt := st.chat_input("Nhap cau hoi de so sanh 2 phuong phap...", key="compare_input"):
        
        # Add placeholder for loading
        with main_container:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                loading_std = st.info("⏳ Dang truy van Standard RAG...")
            with c2:
                loading_graph = st.info("⏳ Dang truy van GraphRAG...")
        
        # Call API
        res = api_post("/api/compare-rag", json_data={
            "query": prompt.strip(), 
            "domain": selected_domain
        })
        
        if res and res.status_code == 200:
            data = res.json()
            std_answer = data.get("standard_rag_answer", "Khong tim thay du lieu.")
            graph_answer = data.get("graph_rag_answer", "Khong tim thay du lieu.")
            
            # Store as a pair
            st.session_state["compare_qa_pairs"].append({
                "question": prompt,
                "standard": std_answer,
                "graph": graph_answer
            })
        else:
            detail = f" ({res.status_code})" if res else ""
            st.session_state["compare_qa_pairs"].append({
                "question": prompt,
                "standard": f"Loi API{detail}",
                "graph": f"Loi API{detail}"
            })
        
        st.rerun()