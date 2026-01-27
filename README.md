# Zero-Shot Record Linkage

Link records across datasets using pre-trained language models. No labeled training data or API keys required.

**Paper**: [Zero-Shot Record Linkage with Ensemble Retrieval and Cross-Encoder Reranking](https://www.dropbox.com/scl/fi/tzvpp2lurejtbw6t4skds/ensemble_linkage.pdf?rlkey=00x7nxbto7d8r44m8igi4ldd1&e=2&st=7zpr8z8k&dl=0)

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Output Format](#output-format)
5. [Hardware Requirements](#hardware-requirements)
6. [Citation](#citation)

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/noahdasanaike/zeroshot_linkage.git
cd zeroshot_linkage
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install PyTorch

Install PyTorch for your system from https://pytorch.org/get-started/locally/

**For GPU (NVIDIA CUDA):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install FAISS

**For GPU:**
```bash
pip install faiss-gpu
```

**For CPU:**
```bash
pip install faiss-cpu
```

### Step 6: Verify installation

```bash
python -c "from zeroshot_linkage import link; print('Installation successful!')"
```

The first run will download the embedding and reranker models (~2GB total). This only happens once.

---

## Quick Start

```python
import pandas as pd
from zeroshot_linkage import link

# Load your datasets
queries = pd.DataFrame({
    "name": ["John Smith", "Jane Doe", "Robert Johnson"]
})

corpus = pd.DataFrame({
    "name": ["J. Smith", "Jane M. Doe", "Bob Wilson", "R. Johnson"]
})

# Link them
results = link(queries, corpus, column_query="name")

print(results)
```

Output:
```
   query_idx      query_text  match_idx   match_text  score
0          0      John Smith          0     J. Smith  0.847
1          1        Jane Doe          1  Jane M. Doe  0.923
2          2  Robert Johnson          3   R. Johnson  0.812
```

---

## Configuration

### Basic Parameters

```python
results = link(
    queries,                # DataFrame: records to match
    corpus,                 # DataFrame: reference records to match against
    column_query="name",    # str: column name in queries
    column_corpus="name",   # str: column name in corpus (defaults to column_query)
)
```

### Threshold

The `threshold` parameter sets the minimum score required to accept a match.

**Default: 0.0** (always return the top candidate)

Our experiments found that the cross-encoder reranker is accurate enough that accepting its top candidate maximizes F1 score. Setting a higher threshold reduces recall without meaningfully improving precision.

```python
# Default: accept the reranker's top candidate (recommended)
results = link(queries, corpus, column_query="name", threshold=0.0)

# Higher threshold: only return high-confidence matches
results = link(queries, corpus, column_query="name", threshold=0.5)
```

Records with scores below the threshold will have `match_idx=None`.

### Retrieval Candidates

The `retrieval_top_k` parameter controls how many candidates are retrieved before reranking.

```python
# Retrieve more candidates (slower but may find better matches)
results = link(queries, corpus, column_query="name", retrieval_top_k=50)

# Retrieve fewer candidates (faster)
results = link(queries, corpus, column_query="name", retrieval_top_k=10)
```

Default is 20, which works well for most cases.

### Embedding Model

The `embedding_model` parameter sets the model used for dense retrieval.

```python
# Default: best quality
results = link(
    queries, corpus, column_query="name",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"
)

# Alternative: faster, smaller, lower quality
results = link(
    queries, corpus, column_query="name",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `Qwen/Qwen3-Embedding-0.6B` | 600MB | Medium | Best |
| `sentence-transformers/all-mpnet-base-v2` | 420MB | Medium | Good |
| `sentence-transformers/all-MiniLM-L6-v2` | 80MB | Fast | Moderate |

### Reranker Model

The `reranker_model` parameter sets the cross-encoder used for scoring candidates.

```python
# Default: multilingual, good quality
results = link(
    queries, corpus, column_query="name",
    reranker_model="jinaai/jina-reranker-v2-base-multilingual"
)

# Alternative: faster, English-only
results = link(
    queries, corpus, column_query="name",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `jinaai/jina-reranker-v2-base-multilingual` | 560MB | Medium | Multilingual support |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | Fast | English only |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 17MB | Fastest | Lower quality |

### Progress Bars

```python
# Disable progress bars (useful for scripts/logging)
results = link(queries, corpus, column_query="name", show_progress=False)
```

### Full Example with All Options

```python
results = link(
    queries,
    corpus,
    column_query="company_name",
    column_corpus="organization_name",
    threshold=0.0,
    retrieval_top_k=30,
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    reranker_model="jinaai/jina-reranker-v2-base-multilingual",
    show_progress=True,
)
```

---

## Output Format

The function returns a pandas DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `query_idx` | int | Row index in the query DataFrame |
| `query_text` | str | The text that was matched |
| `match_idx` | int or None | Row index in the corpus DataFrame |
| `match_text` | str or None | The matched text |
| `score` | float | Confidence score (0-1) |

### Merging Results Back

```python
# Link records
results = link(queries, corpus, column_query="name")

# Merge corpus columns into queries
merged = queries.merge(
    results[["query_idx", "match_idx", "score"]],
    left_index=True,
    right_on="query_idx",
    how="left"
)

# Add corpus data
merged = merged.merge(
    corpus,
    left_on="match_idx",
    right_index=True,
    how="left",
    suffixes=("_query", "_corpus")
)
```

---

## Hardware Requirements

| Setup | RAM | Notes |
|-------|-----|-------|
| Minimum | 8GB | CPU only, slower |
| Recommended | 16GB + GPU | NVIDIA GPU with 6GB+ VRAM |

### Performance Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch size**: Process in batches of 10,000-50,000 queries for large datasets
3. **Smaller models**: Use MiniLM models if speed is critical

---

## Citation

```bibtex
@article{dasanaike2025zeroshot,
  title={Zero-Shot Record Linkage with Ensemble Retrieval and Cross-Encoder Reranking},
  author={Dasanaike, Noah},
  year={2025}
}
```

## License

MIT
