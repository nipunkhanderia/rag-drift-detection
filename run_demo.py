"""
RAG + Drift Detection Demo
============================

Usage:
    python run_demo.py                    # Mock mode (no LLM needed)
    python run_demo.py --model llama3.2   # Uses Ollama with llama3.2

Flow:
1. Build RAG vectorstore from company policy docs
2. Day 1: Ask in-context questions → good scores → snapshot embeddings
3. Day 2: Ask out-of-context questions → weak scores → snapshot query embeddings
4. Drift detector compares Day 1 vs Day 2 → FLAGS drift
5. Generate 2D plot showing the embedding space shift
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_drift import SnapshotStore, DriftDetector
from embedding_drift.viz.plots import plot_2d_drift
from rag_pipeline.rag import build_vectorstore, get_raw_embeddings, ask, embedding_model, init_llm

DRIFT_STORE = ".drift_snapshots"

IN_CONTEXT_QUESTIONS = [
    "How many annual leave days do employees get?",
    "What is the remote work policy?",
    "Who is covered by medical insurance?",
    "How often are performance reviews conducted?",
    "What is the data security policy?",
    "Can unused leave be carried over?",
    "What are the core hours for remote work?",
    "When does medical insurance coverage start?",
]

OUT_OF_CONTEXT_QUESTIONS = [
    "What is the recipe for chocolate cake?",
    "How do I train a neural network?",
    "What are the best stocks to invest in 2024?",
    "Explain quantum computing in simple terms",
    "What is the weather forecast for tomorrow?",
    "How do I fix a leaking faucet?",
    "What are the rules of cricket?",
    "How to build a rocket engine?",
]


def run_questions(db, questions: list[str], label: str) -> np.ndarray:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    query_embeddings = []
    scores = []

    for q in questions:
        q_emb = embedding_model.embed_query(q)
        query_embeddings.append(q_emb)

        result = ask(db, q)
        scores.append(result["relevance_score"])
        status = "PASS" if result["relevance_score"] > 0.7 else "FAIL"
        print(f"  [{status}] [{result['relevance_score']:.3f}] Q: {q}")
        print(f"        A: {result['answer'][:80]}")

    avg_score = np.mean(scores)
    print(f"\n  Average relevance: {avg_score:.3f}")
    return np.array(query_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="RAG + Drift Detection Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model name (e.g. llama3.2). If not provided, uses mock responses.")
    args = parser.parse_args()

    # Initialize LLM (or mock mode)
    init_llm(args.model)
    mode = f"Ollama ({args.model})" if args.model else "Mock mode (no LLM)"

    print("=" * 60)
    print("  RAG + EMBEDDING DRIFT DETECTION DEMO")
    print(f"  Mode: {mode}")
    print("=" * 60)

    # Build vectorstore
    print("\n[1] Building RAG vectorstore from company_policy.txt...")
    db = build_vectorstore()
    doc_embeddings = get_raw_embeddings(db)
    print(f"    Indexed {doc_embeddings.shape[0]} chunks, dim={doc_embeddings.shape[1]}")

    store = SnapshotStore(DRIFT_STORE)

    # Day 1: In-context questions
    day1 = date(2024, 7, 1)
    day1_embeddings = run_questions(db, IN_CONTEXT_QUESTIONS, "DAY 1: In-Context Questions (should work well)")
    store.save(day1_embeddings, snapshot_date=day1)

    # Day 2: Out-of-context questions
    day2 = date(2024, 7, 2)
    day2_embeddings = run_questions(db, OUT_OF_CONTEXT_QUESTIONS, "DAY 2: Out-of-Context Questions (should fail)")
    store.save(day2_embeddings, snapshot_date=day2)

    # Drift detection
    print(f"\n{'='*60}")
    print("  DRIFT DETECTION RESULTS")
    print(f"{'='*60}")

    detector = DriftDetector(thresholds={
        "centroid_shift": 0.3,
        "cosine_drift": 0.01,
        "mmd": 0.03,
        "spread_change": 0.2,
    })

    result = detector.check_drift(day1_embeddings, day2_embeddings)

    print(f"\n  Metrics:")
    for k, v in result["metrics"].items():
        print(f"    {k:<16}: {v:.6f}")

    print(f"\n  DRIFT DETECTED: {'YES' if result['drifted'] else 'NO'}")

    if result["alerts"]:
        print(f"\n  Alerts triggered:")
        for a in result["alerts"]:
            print(f"    >> {a['metric']} = {a['value']:.4f} (threshold: {a['threshold']})")
        print(f"\n  -> Query patterns have shifted significantly!")
        print(f"  -> Users are asking questions outside the knowledge base.")
        print(f"  -> RAG quality is likely degrading.")

    # Visualization
    print(f"\n{'='*60}")
    print("  GENERATING 2D DRIFT VISUALIZATION")
    print(f"{'='*60}")

    plot_2d_drift(store, save_path="drift_visualization.png")
    print("\n  Open drift_visualization.png to see the embedding space shift.")


if __name__ == "__main__":
    main()
