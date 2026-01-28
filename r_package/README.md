# EnsembleLink

Accurate record linkage in R without training data. Uses ensemble retrieval and cross-encoder reranking.

## Installation

```r
# Install from GitHub
devtools::install_github("username/ensemblelink")

# Or install locally
devtools::install("path/to/ensemblelink")
```

### Python Dependencies

The package requires Python with several ML libraries. Install them with:

```r
library(ensemblelink)

# CPU-only (slower but works everywhere)
install_ensemblelink()

# With GPU support (recommended)
install_ensemblelink(gpu = TRUE)
```

Or manually in Python:
```bash
pip install torch sentence-transformers faiss-cpu scikit-learn tqdm
```

## Usage

```r
library(ensemblelink)

# Simple example
queries <- c("New York City", "Los Angelas", "Chcago", "San Fran")
corpus <- c("New York, NY", "Los Angeles, CA", "Chicago, IL",
            "Houston, TX", "San Francisco, CA")

results <- ensemble_link(queries, corpus)
print(results)
#>           query             match
#> 1 New York City      New York, NY
#> 2    Los Angelas   Los Angeles, CA
#> 3        Chcago       Chicago, IL
#> 4      San Fran San Francisco, CA
```

### With Match Scores

```r
results <- ensemble_link(queries, corpus, return_scores = TRUE)
print(results)
#>           query             match     score
#> 1 New York City      New York, NY 0.9823451
#> 2    Los Angelas   Los Angeles, CA 0.9756234
#> 3        Chcago       Chicago, IL 0.9812345
#> 4      San Fran San Francisco, CA 0.9634521
```

### Custom Models

```r
# Use different embedding/reranker models
results <- ensemble_link(
  queries, corpus,
  embedding_model = "BAAI/bge-small-en-v1.5",
  reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### Specifying Python Environment

```r
# Use a specific conda environment
configure_python(condaenv = "myenv")

# Or a specific Python installation
configure_python(python = "/path/to/python")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `queries` | (required) | Character vector of strings to match |
| `corpus` | (required) | Character vector of reference strings |
| `embedding_model` | "Qwen/Qwen3-Embedding-0.6B" | Sentence-transformers model for embeddings |
| `reranker_model` | "jinaai/jina-reranker-v2-base-multilingual" | Cross-encoder for reranking |
| `top_k` | 30 | Candidates to retrieve before reranking |
| `return_scores` | FALSE | Return match confidence scores |
| `show_progress` | TRUE | Show progress bar |
| `device` | "auto" | "cuda", "cpu", or "auto" |

## How It Works

1. **Ensemble Retrieval**: Combines dense semantic embeddings (FAISS) with sparse character n-grams (TF-IDF) to find candidate matches
2. **Cross-Encoder Reranking**: Jointly scores query-candidate pairs using a transformer model
3. **Top-1 Selection**: Returns the highest-scoring candidate

The method requires no labeled training data and outperforms supervised approaches on standard benchmarks.

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
