import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from source.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
# Thư mục lưu Vector Database của RAG truyền thống
FAISS_DIR = Path("./faiss_index")

# Tái sử dụng model từ file .env của bạn
# embeddings = OpenAIEmbeddings(
#     openai_api_key=settings.embedding_api_key,
#     openai_api_base=settings.embedding_base_url,
#     model=settings.embedding_model
# )

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
llm = ChatOpenAI(
    openai_api_key=settings.llm_api_key,
    openai_api_base=settings.llm_base_url,
    model_name=settings.llm_model,
    max_tokens=1500
)

def build_faiss_index(txt_path: Path):
    """Băm nhỏ file TXT và lưu vào FAISS (Vector DB)"""
    text = txt_path.read_text(encoding="utf-8")
    
    # Cắt văn bản thành các đoạn 1000 ký tự
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    
    # Nếu đã có DB cũ thì nạp thêm vào, chưa có thì tạo mới
    if FAISS_DIR.exists():
        vector_db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        vector_db.add_documents(chunks)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)
        
    # Lưu xuống ổ cứng (Sẽ tự tạo 2 file: index.faiss và index.pkl)
    vector_db.save_local(str(FAISS_DIR))
    return True

def query_standard_rag(question: str):
    """Hỏi đáp với RAG truyền thống dùng kiến trúc LCEL mới"""
    if not FAISS_DIR.exists():
        return "Chưa có dữ liệu FAISS. Vui lòng upload tài liệu trước."
        
    vector_db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Tạo mẫu câu hỏi ép AI trả lời dựa trên văn bản
    template = """Dựa vào các thông tin sau đây để trả lời câu hỏi. Nếu không tìm thấy câu trả lời trong thông tin này, hãy nói là bạn không biết, đừng tự bịa ra câu trả lời.
    
    Thông tin:
    {context}
    
    Câu hỏi: {question}
    Trả lời:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Chuỗi LCEL hiện đại (thay thế cho RetrievalQA cũ)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)