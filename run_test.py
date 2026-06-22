"""
Test drift detection with custom question files.

Usage:
    python run_test.py good_questions.txt          # Should show NO drift
    python run_test.py bad_questions.txt           # Should show drift detected
    python run_test.py good_questions.txt --model llama3.2  # With Ollama
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

DRIFT_STORE = ".drift_snapshots_test"


def load_questions(filepath: str) -> list[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


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
    parser = argparse.ArgumentParser(description="Test drift detection with custom questions")
    parser.add_argument("questions_file", type=str, help="Path to questions file (one question per line)")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name. If not provided, uses mock mode.")
    args = parser.parse_args()

    if not Path(args.questions_file).exists():
        print(f"Error: File '{args.questions_file}' not found.")
        sys.exit(1)

    init_llm(args.model)
    mode = f"Ollama ({args.model})" if args.model else "Mock mode (no LLM)"

    print("=" * 60)
    print("  DRIFT DETECTION TEST")
    print(f"  Mode: {mode}")
    print(f"  Questions file: {args.questions_file}")
    print("=" * 60)

    # Build vectorstore
    print("\n[1] Building RAG vectorstore from company_policy.txt...")
    db = build_vectorstore()
    doc_embeddings = get_raw_embeddings(db)
    print(f"    Indexed {doc_embeddings.shape[0]} chunks, dim={doc_embeddings.shape[1]}")

    store = SnapshotStore(DRIFT_STORE)

    # Baseline: in-context questions (always use these as Day 1)
    baseline_questions = [
        "How many annual leave days do employees get?",
        "What is the remote work policy?",
        "Who is covered by medical insurance?",
        "How often are performance reviews conducted?",
        "What is the data security policy?",
        "Can unused leave be carried over?",
        "What are the core hours for remote work?",
        "When does medical insurance coverage start?",
    ]

    day1 = date(2024, 7, 1)
    day1_embeddings = run_questions(db, baseline_questions, "BASELINE (Day 1): In-Context Questions")
    store.save(day1_embeddings, snapshot_date=day1)

    # Test: user-provided questions (Day 2)
    test_questions = load_questions(args.questions_file)
    day2 = date(2024, 7, 2)
    day2_embeddings = run_questions(db, test_questions, f"TEST (Day 2): {args.questions_file}")
    store.save(day2_embeddings, snapshot_date=day2)

    # Drift detection
    print(f"\n{'='*60}")
    print("  DRIFT DETECTION RESULTS")
    print(f"{'='*60}")

    detector = DriftDetector(thresholds={
        "centroid_shift": 0.5,
        "cosine_drift": 0.5,
        "mmd": 0.2,
        "spread_change": 0.3,
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

    # Visualization
    output_name = Path(args.questions_file).stem + "_drift.png"
    print(f"\n{'='*60}")
    print("  GENERATING 2D DRIFT VISUALIZATION")
    print(f"{'='*60}")

    plot_2d_drift(store, save_path=output_name)
    print(f"\n  Open {output_name} to see the embedding space shift.")


if __name__ == "__main__":
    main()
