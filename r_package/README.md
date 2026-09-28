# EnsembleLink

Accurate record linkage in R without training data. Uses ensemble retrieval and four-expert agreement fusion.

## Installation

```r
# Install from GitHub
devtools::install_github("noahdasanaike/ensemblelink/r_package")

# Or install locally
devtools::install("path/to/ensemblelink")
```

### Python Dependencies

The package requires Python with several ML libraries. Install them with:

```r
library(ensemblelink)

install_ensemblelink()
```

Or manually in Python:
```bash
pip install torch "transformers<5" "sentence-transformers>=2.7" rapidfuzz scikit-learn scipy Unidecode tqdm einops
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
# adds match_index, score (the confidence), margin and reranker_probability
```

`score` is the paper's confidence (rule B2): the mean of two percentile ranks among the queries linked in the same call, the rank of the fused top-minus-runner-up margin and the rank of the match's mean reranker probability; 0 when the top fused score is tied. It orders the matches of one call from least to most trustworthy and does not change which record is matched. It is relative to the call, not a probability: a threshold chosen on one set of queries carries over to another call only when the query sets are alike, and linking a batch in pieces gives different scores than linking it at once. `reranker_probability` is an absolute score.

### Custom Models

```r
# Other models (rerankers: single-logit sequence-classification cross-encoders)
results <- ensemble_link(
  queries, corpus,
  embedding_model = "BAAI/bge-small-en-v1.5",
  reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### Multi-Column Matching

Write each record as `field=value` pairs joined by `" | "` (omit missing fields), and set `multifield = TRUE`, which adds the paper's embedding instruction for multi-field records:

```r
queries <- paste0("city=", df$city, " | state=", df$state)
corpus  <- paste0("city=", ref$city, " | state=", ref$state)
results <- ensemble_link(queries, corpus, multifield = TRUE)
```

This is the field-count rule of the Python package (`link(..., columns_query=[...])`), which builds the text for you.

### Hierarchical Blocking

Use `ensemble_link_blocked()` when you need to match on multiple levels - for example, matching states first, then counties within matched states:

```r
# Query data
query_states <- c("Kalifornia", "Texass", "New Yrok")
query_counties <- c("Los Angelos", "Harris Co", "Queens County")

# Corpus data
corpus_states <- c("California", "California", "Texas", "Texas", "New York")
corpus_counties <- c("Los Angeles", "San Francisco", "Harris", "Dallas", "Queens")

# Match with blocking
results <- ensemble_link_blocked(
  query_blocks = query_states,
  query_details = query_counties,
  corpus_blocks = corpus_states,
  corpus_details = corpus_counties,
  return_scores = TRUE
)

print(results)
#>   query_block   query_detail match_block match_detail match_index block_score detail_score
#> 1  Kalifornia    Los Angelos  California  Los Angeles           1       0.892        0.945
#> 2      Texass      Harris Co       Texas       Harris           3       0.876        0.823
#> 3    New Yrok  Queens County    New York       Queens           5       0.834        0.891
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
| `embedding_model` | "microsoft/harrier-oss-v1-0.6b" | Sentence-transformers model for embeddings |
| `reranker_model` | "jinaai/jina-reranker-v2-base-multilingual" | First reranker (CC-BY-NC-4.0; NULL to drop) |
| `reranker_model_2` | "BAAI/bge-reranker-v2-m3" | Second reranker (Apache-2.0; NULL to drop) |
| `reranker_model_3` | NULL | Optional third reranker, e.g. "zeroentropy/zerank-2-reranker" |
| `pool_size` | 30 | Candidates retrieved from each of dense and sparse retrieval |
| `multifield` | FALSE | Records are `field=value` pairs: add the multi-field embedding instruction |
| `return_scores` | FALSE | Return `match_index`, `score`, `margin`, `reranker_probability` |
| `show_progress` | TRUE | Show progress bar |
| `device` | "auto" | "cuda", "cpu", or "auto" |
| `exact` | TRUE | TRUE: the benchmark's batching (reproduces the paper). FALSE: each distinct query, record and pair scored once in length-sorted batches (faster with duplicates; bfloat16 scores differ at noise level) |
| `index_cache` | NULL | Directory for an on-disk cache of the corpus index; a later call with the same corpus and settings loads it instead of re-embedding. Stored with Python's pickle: use a trusted directory |

Licenses: the default set includes Jina v2 (CC-BY-NC-4.0) and is for non-commercial use. For commercial use, `reranker_model = NULL, reranker_model_3 = "zeroentropy/zerank-2-reranker"` keeps only Apache-2.0 and MIT models; it scored below the default in the paper's development tests (see the main README).

## How It Works

The R package runs the Python package's code (vendored in `inst/python/ensemblelink_py`), so results match `zeroshot_linkage` in Python. For each query, EnsembleLink retrieves a candidate pool (the union of the 30 nearest records by embedding cosine, exact search, and the 30 nearest by character TF-IDF on transliterated text; 60 by embedding when the query shares no character n-gram with the corpus) and scores every candidate with four experts:

1. **Reranker ensemble**: Jina v2 and BGE v2-m3 read the query and candidate together; their per-pool z-scores are summed.
2. **CSLS-corrected dense cosine**: embedding cosine with a hubness penalty.
3. **Sparse TF-IDF cosine** on transliterated text.
4. **Jaro-Winkler** on transliterated text.

The experts are fused **without labeled data**: each expert's weight is the squared share of queries on which its top pick agrees with the consensus (an expert with no spread over a pool abstains). On a GPU the models run in bfloat16, as in the paper.

## Citation

```bibtex
@unpublished{dasanaike2026zeroshot,
  title  = {Pre-Trained Language Models as Zero-Shot Tools for Social Science Research},
  author = {Dasanaike, Noah},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT
