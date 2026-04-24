from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rag.citation import CitationMapper


def main() -> None:

    try:
        from rag.singletons import initialize_all
        components = initialize_all()
        embedder = components["embedder"]
        retriever = components["retriever"]
        context_builder = components["context_builder"]
        generator = components["generator"]
        token_manager = components["token_manager"]
    except Exception as e:
        if "does not exist" in str(e):
            print("\n[LỖI] Collection 'rag_documents' chưa tồn tại.")
            print(">>> Chạy lại populate_db.py")
        else:
            print(f"[INIT ERROR] {e}")
        return

    print("--- RAG Chatbot sẵn sàng | 'Ctrl + c' để thoát ---")

    while True:
        query = input("\nHãy đặt câu hỏi cho tôi tại đây: ").strip()

        if query.lower() in ["quit", "exit"]:
            print("Pái Paiiiii!")
            break

        if not query:
            continue

        print("Đang xử lý...")

        try:
            query_embedding = embedder.embed_text(query)
            if not query_embedding or sum(query_embedding) == 0:
                raise ValueError("Embedding không hợp lệ")
        except Exception as e:
            print(f"Lỗi embedding: {e}")
            continue

        try:
            retrieved_docs = retriever.retrieve(query_embedding, k=8)
            documents = retrieved_docs.get("documents")

            if not documents or not documents[0]:
                print("AI: Hiện tại tôi chưa thể trả lời câu hỏi này, Bạn vui lòng liên hệ với Account hoặc TSM/MKT để được giải đáp thắc mắc. Xin cảm ơn")
                continue
        except Exception as e:
            print(f"Lỗi Retrieve: {e}")
            continue

        try:
            context = context_builder.build(retrieved_docs)
        except Exception as e:
            print(f"Lỗi Context build: {e}")
            continue

        selected_model, _ = token_manager.select_model(query)

        try:
            generated_response, token_info = generator.generate(
                query,
                context,
                model_name=selected_model,
            )
            raw_answer = generated_response.get(
                "answer",
                "Hiện tại tôi chưa thể trả lời câu hỏi này, Bạn vui lòng liên hệ với Account hoặc TSM/MKT để được giải đáp thắc mắc. Xin cảm ơn",
            )
        except Exception as e:
            print(f"Lỗi Generate: {e}")
            continue

        try:
            metadatas = retrieved_docs.get("metadatas", [[]])[0]
            mapper = CitationMapper(metadatas)
            final_answer = mapper.replace(raw_answer)
        except Exception as e:
            print(f"Lỗi Citation mapping: {e}")
            final_answer = raw_answer
            mapper = None

        print("\nAI:\n", final_answer)

        if mapper:
            try:
                mapper.print_citations()
            except Exception:
                pass

    try:
        token_manager.save_session_log()
    except Exception:
        pass


if __name__ == "__main__":
    main()