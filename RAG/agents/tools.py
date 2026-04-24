from rag.embedding import GeminiEmbedder
from rag.retriever import Retriever
from rag.context_builder import ContextBuilder


embedder = GeminiEmbedder()
retriever = Retriever()
context_builder = ContextBuilder()


def rag_search(question):

    query_embedding = embedder.embed_text(question)

    results = retriever.retrieve(query_embedding)

    context = context_builder.build(results)

    return context


def calculator(expression):

    try:
        return str(eval(expression))
    except:
        return "Error"