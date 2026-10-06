# EnsembleLink

Record linkage in R without training data, using pre-trained language models: ensemble retrieval and four-expert agreement fusion. The R package runs the code of the Python package (vendored in `inst/python/ensemblelink_py`), so the two give the same results.

## Installation

```r
# Install from GitHub
devtools::install_github("noahdasanaike/ensemblelink/r_package")

# Or install locally
devtools::install("path/to/ensemblelink")
```

### Python dependencies

The package needs Python with several machine-learning libraries. Install them from R, once, and then restart R:

```r
library(ensemblelink)
install_ensemblelink()
```

Or install them in Python directly:

```bash
pip install torch "transformers<5" "sentence-transformers>=2.7" rapidfuzz scikit-learn scipy Unidecode tqdm einops
```

If you have a GPU, install a CUDA build of PyTorch (https://pytorch.org/get-started/locally/). The first run downloads the embedding model and the two rerankers (about 4 GB).

### Choosing a Python environment

```r
configure_python(condaenv = "myenv")          # a conda environment
configure_python(python = "/path/to/python")  # or a specific Python installation
```

## Usage

```r
library(ensemblelink)

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

### Match scores

```r
results <- ensemble_link(queries, corpus, return_scores = TRUE)
# adds match_index, score (the confidence), margin and reranker_probability
```

`score` is, for each query, the mean of two percentile ranks among the queries linked in the same call: the rank of the fused top-minus-runner-up margin and the rank of the match's mean reranker probability. It is 0 when the top fused score is tied. The score orders the matches of one call from least to most trustworthy and does not change which record is matched. Because it is relative to the call, it is not a probability: a threshold chosen on one set of queries carries over to another call only when the query sets are alike, and linking a batch in pieces gives different scores than linking it at once. `reranker_probability` is an absolute score, comparable across calls.

### Several fields

Write each record as `field=value` pairs joined by `" | "` (omit missing fields), and set `multifield = TRUE`, which adds the embedding instruction for records with several fields:

```r
queries <- paste0("city=", df$city, " | state=", df$state)
corpus  <- paste0("city=", ref$city, " | state=", ref$state)
results <- ensemble_link(queries, corpus, multifield = TRUE)
```

The Python package builds this text for you (`link(..., columns_query=[...])`).

### Hierarchical matching

Use `ensemble_link_blocked()` to match on two levels, for example states first and then counties within the matched states:

```r
query_states <- c("Kalifornia", "Texass", "New Yrok")
query_counties <- c("Los Angelos", "Harris Co", "Queens County")

corpus_states <- c("California", "California", "Texas", "Texas", "New York")
corpus_counties <- c("Los Angeles", "San Francisco", "Harris", "Dallas", "Queens")

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
| `show_progress` | TRUE | Show a progress bar |
| `device` | "auto" | "cuda", "cpu", or "auto" |
| `exact` | TRUE | TRUE: fixed batches (deterministic for a given GPU). FALSE: each distinct query, record and pair scored once in length-sorted batches (faster when texts repeat; bfloat16 scores differ at noise level) |
| `index_cache` | NULL | Directory for an on-disk cache of the corpus index; a later call with the same corpus and settings loads it instead of re-embedding. Stored with Python's pickle, so use a trusted directory |

Other models: any sentence-transformers embedding model, and rerankers that are sequence-classification cross-encoders with a single relevance logit.

```r
results <- ensemble_link(
  queries, corpus,
  embedding_model = "BAAI/bge-small-en-v1.5",
  reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

## Speed

On one NVIDIA A100 (80 GB), linking 1,000 five-field voter records against a corpus of one million records takes about 10 minutes, most of it spent embedding and indexing the corpus. Use `index_cache` to embed a corpus once and reuse it across calls. A GPU is strongly recommended; on CPU the models run in float32 and are much slower. On a GPU they run in bfloat16.

## Models and licenses

| Model | Role | Parameters | License |
|---|---|---:|---|
| `microsoft/harrier-oss-v1-0.6b` | embedding | 0.6B | MIT |
| `jinaai/jina-reranker-v2-base-multilingual` | reranker (default) | 278M | CC-BY-NC-4.0 (non-commercial) |
| `BAAI/bge-reranker-v2-m3` | reranker (default) | 568M | Apache-2.0 |
| `zeroentropy/zerank-2-reranker` | reranker (optional) | 4.0B | Apache-2.0 |

Because of Jina v2, the default model set is for research and other non-commercial use. For commercial use, `reranker_model = NULL, reranker_model_3 = "zeroentropy/zerank-2-reranker"` keeps only Apache-2.0 and MIT models; zerank-2 is considerably slower.

## How it works

For each query, EnsembleLink retrieves a candidate pool: the union of the 30 nearest records by embedding cosine (exact search) and the 30 nearest by character TF-IDF on transliterated text, or the 60 nearest by embedding when the query shares no character n-gram with the corpus. It then scores every candidate with four experts:

1. **Reranker ensemble**: Jina v2 and BGE v2-m3 read the query and candidate together; their per-pool z-scores are summed.
2. **CSLS-corrected dense cosine**: embedding cosine with a hubness penalty.
3. **Sparse TF-IDF cosine** on transliterated text.
4. **Jaro-Winkler** on transliterated text.

The experts are fused **without labeled data**: each expert's weight is the squared share of queries on which its top pick agrees with the consensus, and an expert that gives every candidate in a pool the same score abstains.

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
