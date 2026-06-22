# RAG + Embedding Drift Detection

Two components in one project:

1. **embedding_drift** — pip-installable package that snapshots embeddings daily and detects drift
2. **rag_pipeline** — simple RAG system using Ollama llama3.2 (or mock mode)

## How It Works

- RAG answers questions using company policy docs
- Each day, query embeddings are snapshotted
- When users start asking out-of-context questions, the query embeddings shift
- Drift detector flags this shift before you notice performance degradation

### How Drift is Introduced

- Day 1: Users ask in-context HR questions → query embeddings cluster near policy topics
- Day 2: Users ask random out-of-context questions → embeddings land in completely different regions
- The shift between these two clusters = **drift**

### How Drift is Caught

4 metrics compare Day 1 vs Day 2 query embeddings:

| Metric | What it catches |
|--------|----------------|
| `centroid_shift` | Center of mass of queries moved |
| `cosine_drift` | Direction of average query changed |
| `mmd` | Overall distribution shape changed |
| `spread_change` | Queries became more scattered |

When values exceed thresholds → alerts fire.

## Setup

```bash
# Prerequisites: Ollama running with llama3.2 (only if using --model)
ollama pull llama3.2

# Install
cd rag_drift_project
pip install -e ".[rag]"
# OR
pip install -r requirements.txt
pip install -e .
```

## Run

### Mock mode (no LLM needed, uses sample answers)

```bash
python run_demo.py
```

Uses pre-built answers for the test questions. Good for testing the drift detection pipeline without needing Ollama running.

### With Ollama (real LLM inference)

```bash
python run_demo.py --model llama3.2
```

Uses llama3.2 via Ollama for live inference. You can pass any Ollama model name.

### What happens when you run

1. Builds a FAISS vectorstore from company policy docs
2. Day 1: asks 8 in-context HR questions → high relevance scores → snapshots query embeddings
3. Day 2: asks 8 out-of-context questions → low relevance scores → snapshots query embeddings
4. Compares Day 1 vs Day 2 embeddings → **drift detected!**
5. Generates `drift_visualization.png` showing the 2D embedding space shift

## Project Structure

```
rag_drift_project/
├── embedding_drift/        # pip-installable drift detection package
│   ├── core/
│   │   ├── snapshot.py     # Save/load daily embedding snapshots
│   │   └── drift.py        # Drift metrics (centroid, cosine, MMD, spread)
│   └── viz/
│       └── plots.py        # 2D PCA visualization
├── rag_pipeline/           # RAG system
│   ├── data/
│   │   └── company_policy.txt
│   └── rag.py              # FAISS + Ollama/mock mode
├── run_demo.py             # Main demo script
├── pyproject.toml          # Package config
└── requirements.txt
```
