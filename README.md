# EnsembleLink

Accurate record linkage without training data using pre-trained language models. No labeled examples or API keys required. Bug reports welcome.

**Paper**: [EnsembleLink: Accurate Record Linkage Without Training Data](https://www.dropbox.com/scl/fi/tzvpp2lurejtbw6t4skds/ensemble_linkage.pdf?rlkey=00x7nxbto7d8r44m8igi4ldd1&e=2&st=7zpr8z8k&dl=0)

## How It Works

For each query, EnsembleLink retrieves a candidate pool (the union of the top dense-embedding and top character-n-gram TF-IDF matches) and scores every candidate with four complementary experts:

1. **Reranker ensemble** — two cross-encoders (Jina v2 and BGE v2-m3) read the query and candidate together; their per-pool z-scores are summed.
2. **CSLS-corrected dense cosine** — embedding cosine with a hubness penalty (`2 * cosine - hubness`) so that corpus points near everything are discounted.
3. **Sparse TF-IDF cosine** — character-n-gram overlap, which catches abbreviations and typos.
4. **Jaro-Winkler** — a classical lexical edit similarity.

Each expert is z-normalized within the candidate pool, and the experts are fused **without any labeled data**: every expert votes for its top candidate per query, and each expert's weight is the squared fraction of queries on which it agrees with the consensus. The weighted sum of the experts decides the match. Experts that track the crowd are trusted; experts that disagree are discounted.

## Table of Contents

1. [R Package](#r-package)
2. [Python Package](#python-package)
3. [Configuration](#configuration)
4. [Hardware Requirements](#hardware-requirements)
5. [Citation](#citation)

---

## R Package

### Installation

```r
# Install from GitHub
devtools::install_github("noahdasanaike/ensemblelink/r_package")
```

Then install the Python dependencies (one-time setup):

```r
library(ensemblelink)

# For CPU only (most laptop use-cases)
install_ensemblelink(gpu = FALSE)

# For GPU (recommended, much faster)
install_ensemblelink(gpu = TRUE)
```

After installing, **restart R** before using the package.

### Quick Start

```r
library(ensemblelink)

# Your data
queries <- c("John Smith", "Jane Doe", "Robert Johnson")
corpus <- c("J. Smith", "Jane M. Doe", "Bob Wilson", "R. Johnson")

# Link them
results <- ensemble_link(queries, corpus)
print(results)
#>            query       match
#> 1     John Smith    J. Smith
#> 2       Jane Doe Jane M. Doe
#> 3 Robert Johnson  R. Johnson
```

### With Match Scores

```r
results <- ensemble_link(queries, corpus, return_scores = TRUE)
print(results)
#>            query       match     score
#> 1     John Smith    J. Smith 0.7098244
#> 2       Jane Doe Jane M. Doe 0.8300437
#> 3 Robert Johnson  R. Johnson 0.4683043
```

### With Data Frames

```r
# Load your data
queries_df <- read.csv("queries.csv")
corpus_df <- read.csv("corpus.csv")

# Link using specific columns
results <- ensemble_link(
  queries = queries_df$name,
  corpus = corpus_df$organization_name
)

# Add results back to your data
queries_df$matched_name <- results$match
queries_df$match_score <- results$score
```

### R Configuration Options

```r
results <- ensemble_link(
  queries,
  corpus,
  embedding_model = "microsoft/harrier-oss-v1-0.6b",
  reranker_model = "jinaai/jina-reranker-v2-base-multilingual",
  reranker_model_2 = "BAAI/bge-reranker-v2-m3",  # set NULL for a single reranker
  pool_size = 50,
  return_scores = TRUE,
  show_progress = TRUE,
  device = "auto"  # "cuda", "cpu", or "auto"
)
```

### Specifying Python Environment

```r
# Use a specific conda environment
configure_python(condaenv = "myenv")

# Or a specific Python path
configure_python(python = "/path/to/python")

# Then run your linkage
results <- ensemble_link(queries, corpus)
```

### Hierarchical Blocking (R)

Use `ensemble_link_blocked()` when you need to match on multiple levels - for example, matching states first, then counties within matched states:

```r
# Match states first, then counties within states
results <- ensemble_link_blocked(
  query_blocks = c("Kalifornia", "Texass"),
  query_details = c("Los Angelos", "Harris Co"),
  corpus_blocks = c("California", "California", "Texas", "Texas"),
  corpus_details = c("Los Angeles", "San Francisco", "Harris", "Dallas"),
  return_scores = TRUE
)

print(results)
#>   query_block query_detail match_block match_detail match_index block_score detail_score
#> 1  Kalifornia  Los Angelos  California  Los Angeles           1       0.892        0.945
#> 2      Texass    Harris Co       Texas       Harris           3       0.876        0.823
```

---

## Python Package

### Installation

#### Option A: Install with pip (recommended)

```bash
pip install git+https://github.com/noahdasanaike/ensemblelink.git
```

Or clone and install locally:

```bash
git clone https://github.com/noahdasanaike/ensemblelink.git
cd ensemblelink
pip install -e .
```

#### Option B: Manual installation

If you need more control over dependencies (e.g., GPU support):

```bash
git clone https://github.com/noahdasanaike/ensemblelink.git
cd ensemblelink

# Install PyTorch for your system from https://pytorch.org/get-started/locally/
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
# For CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e .
```

#### Verify installation

```bash
python -c "from zeroshot_linkage import link; print('Installation successful!')"
```

The first run will download the embedding model and the two rerankers (~3GB total). This only happens once.

### Quick Start

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

### Python Output Format

The function returns a pandas DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `query_idx` | int | Row index in the query DataFrame |
| `query_text` | str | The text that was matched |
| `match_idx` | int | Row index in the corpus DataFrame |
| `match_text` | str | The matched text |
| `score` | float | Fused match score (higher is better) |

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

### Hierarchical Blocking

Use `link_blocked` when you need to match on multiple levels - for example, matching states first, then counties within matched states:

```python
from zeroshot_linkage import link_blocked

queries = pd.DataFrame({
    "state": ["Kalifornia", "Texass", "New Yrok"],
    "county": ["Los Angelos", "Harris Co", "Queens County"]
})

corpus = pd.DataFrame({
    "state": ["California", "California", "Texas", "Texas", "New York"],
    "county": ["Los Angeles", "San Francisco", "Harris", "Dallas", "Queens"]
})

results = link_blocked(
    queries, corpus,
    blocking_query="state",
    detail_query="county"
)

print(results)
```

Output:
```
   query_idx query_block   query_detail  match_idx match_block  match_detail  block_score  detail_score
0          0  Kalifornia    Los Angelos          0  California   Los Angeles        0.892         0.945
1          1      Texass      Harris Co          2       Texas        Harris        0.876         0.823
2          2    New Yrok  Queens County          4    New York        Queens        0.834         0.891
```

### Custom Model Cache Directory

By default, models are downloaded to `~/.cache/huggingface/`. To use a custom directory:

```python
results = link(
    queries, corpus,
    column_query="name",
    cache_dir="/path/to/models"
)
```

---

## Configuration

### Embedding Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `microsoft/harrier-oss-v1-0.6b` | 600MB | Medium | Best (default) |
| `Qwen/Qwen3-Embedding-0.6B` | 600MB | Medium | Strong |
| `sentence-transformers/all-mpnet-base-v2` | 420MB | Medium | Good |
| `sentence-transformers/all-MiniLM-L6-v2` | 80MB | Fast | Moderate |

### Reranker Models

The reranker expert is an **ensemble of two** cross-encoders by default; their z-scored ranks are summed. Pass a single `reranker_model` (Python) or set `reranker_model_2 = NULL` (R) to use just one.

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `jinaai/jina-reranker-v2-base-multilingual` | 278MB | Medium | Default reranker 1 |
| `BAAI/bge-reranker-v2-m3` | 568MB | Medium | Default reranker 2 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | Fast | English only, lighter |

### Retrieval Candidates

The `pool_size` parameter controls how many candidates are retrieved from **each** of dense and sparse retrieval; the pool is their union. Default is 50.

- **Higher values** (100+): Better recall, slower
- **Lower values** (20): Faster, may miss matches

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
@article{dasanaike2026ensemblelink,
  title={EnsembleLink: Accurate Record Linkage Without Training Data},
  author={Dasanaike, Noah},
  year={2026}
}
```

## License

MIT
