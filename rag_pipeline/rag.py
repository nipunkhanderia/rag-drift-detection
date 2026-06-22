"""RAG pipeline - uses Ollama if model provided, otherwise mock responses."""

import numpy as np
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = Path(__file__).parent / "data" / "company_policy.txt"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample answers for mock mode (no LLM needed)
MOCK_ANSWERS = {
    "How many annual leave days do employees get?": "All full-time employees are entitled to 25 days of paid annual leave per year.",
    "What is the remote work policy?": "Employees may work remotely up to 3 days per week, agreed upon with your team lead.",
    "Who is covered by medical insurance?": "The company provides medical insurance coverage for all employees and their dependents.",
    "How often are performance reviews conducted?": "Performance reviews are conducted quarterly.",
    "What is the data security policy?": "All employees must complete annual data security training. Two-factor authentication is mandatory.",
    "Can unused leave be carried over?": "Unused leave can be carried over up to a maximum of 5 days into the next calendar year.",
    "What are the core hours for remote work?": "All employees must be available during core hours 10am-4pm regardless of location.",
    "When does medical insurance coverage start?": "New employees are enrolled after completing their probation period of 3 months.",
    # Out-of-context → "I don't know"
    "What is the recipe for chocolate cake?": "I don't know.",
    "How do I train a neural network?": "I don't know.",
    "What are the best stocks to invest in 2024?": "I don't know.",
    "Explain quantum computing in simple terms": "I don't know.",
    "What is the weather forecast for tomorrow?": "I don't know.",
    "How do I fix a leaking faucet?": "I don't know.",
    "What are the rules of cricket?": "I don't know.",
    "How to build a rocket engine?": "I don't know.",
}


_llm = None


def init_llm(model_name: str | None = None):
    """Initialize LLM if model provided, else use mock mode."""
    global _llm
    if model_name:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(model=model_name, temperature=0.0)
    else:
        _llm = None


def build_vectorstore():
    loader = TextLoader(str(DATA_PATH))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding_model)


def get_raw_embeddings(db) -> np.ndarray:
    index = db.index
    n = index.ntotal
    vectors = np.zeros((n, index.d), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    return vectors


def ask(db, question: str) -> dict:
    """Ask a question. Uses LLM if initialized, otherwise mock answers."""
    retriever = db.as_retriever(search_kwargs={"k": 3})
    retrieved = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved]

    if _llm:
        # Real LLM mode
        prompt = f"""You are an HR assistant. Answer ONLY using the provided context.
If the answer is not in the context, respond exactly with: I don't know.

Context:
{chr(10).join(contexts)}

Question: {question}

Answer:"""
        response = _llm.invoke(prompt)
        answer = response.content.strip()
    else:
        # Mock mode - use pre-built answers
        answer = MOCK_ANSWERS.get(question, "I don't know.")

    # Relevance score
    answer_emb = embedding_model.embed_query(answer)
    context_emb = embedding_model.embed_query(" ".join(contexts))
    cos_sim = np.dot(answer_emb, context_emb) / (
        np.linalg.norm(answer_emb) * np.linalg.norm(context_emb) + 1e-10
    )

    return {"answer": answer, "contexts": contexts, "relevance_score": float(cos_sim)}
