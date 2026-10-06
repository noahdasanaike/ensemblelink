# EnsembleLink

Record linkage without training data, using pre-trained language models. No labeled examples or API keys are required. Bug reports are welcome.

EnsembleLink is described in Noah Dasanaike, [Pre-Trained Language Models as Zero-Shot Tools for Social Science Research](https://www.dropbox.com/scl/fi/2kx17wvydhr3v8m4u59ek/zeroshot_llms_dasanaike.pdf?rlkey=kybmjjnptca1urqcin06ro349&st=fk6w43uj&e=1&dl=0).

## How it works

For each query, EnsembleLink retrieves a pool of candidate records and then scores every candidate with four experts.

**Retrieval.** The pool is the union of the 30 corpus records closest by embedding cosine (exact search over the whole corpus with `microsoft/harrier-oss-v1-0.6b`) and the 30 closest by character 2-4-gram TF-IDF on transliterated text. A query that shares no character n-gram with any record, such as a name in another script, gets the 60 closest records by embedding instead.

**Experts.**

1. **Reranker ensemble**: two cross-encoders, Jina Reranker v2 (multilingual) and BGE Reranker v2-m3, read the query and candidate together; the expert is the sum of their per-pool z-scores.
2. **CSLS-corrected dense cosine**: `2 * cosine - hubness`, where a record's hubness is its mean cosine with its 10 nearest queries, so that records close to everything are discounted.
3. **Sparse TF-IDF cosine** on transliterated text (Unidecode), which catches abbreviations and typos.
4. **Jaro-Winkler** similarity on transliterated text.

**Fusion without labels.** Each expert is z-scored within the pool. Each expert votes for its top candidate, an expert that gives every candidate in a pool the same score abstains, and each expert's weight is the squared share of queries on which it agrees with the consensus vote. The match is the candidate with the highest weighted sum.

**Record text.** A record with several fields is written as `field=value | field=value` (missing fields omitted), and its query gets the embedding instruction "Retrieve the record that refers to the same entity". A record with a single field is written as its value alone, with no instruction.

## Installation

### Python

```bash
pip install git+https://github.com/noahdasanaike/ensemblelink.git
```

If you have a GPU, install a CUDA build of PyTorch first (https://pytorch.org/get-started/locally/). The first run downloads the embedding model and the two rerankers (about 4 GB).

### R

```r
devtools::install_github("noahdasanaike/ensemblelink/r_package")
library(ensemblelink)
install_ensemblelink()   # installs the Python dependencies, once; then restart R
```

The R package runs the Python package's code (a vendored copy in `inst/python`), so the two give the same results. To use a particular Python, call `configure_python(condaenv = "myenv")` or `configure_python(python = "/path/to/python")`. See [`r_package/README.md`](r_package/README.md) for the full R interface.

## Usage

### Python

```python
import pandas as pd
from zeroshot_linkage import link

queries = pd.DataFrame({"name": ["John Smith", "Jane Doe", "Robert Johnson"]})
corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson", "R. Johnson"]})

results = link(queries, corpus, column_query="name")
```

- **Several fields:** `link(queries, corpus, columns_query=["first", "last", "birth_year"])`. Corpus columns default to the same names, or pass `columns_corpus`, paired by position. Store whole numbers as integers (pandas `Int64`) so that they read `1980`, not `1980.0`.
- **Ties:** pass `id_query` and `id_corpus` (columns of unique IDs) to break exact ties by record ID; otherwise row positions are used.
- **Hierarchical matching:** `link_blocked(queries, corpus, blocking_query="state", detail_query="county")` matches a coarse field first and then the detail within the matched block.
- **Repeated calls:** `FusionMatcher` keeps the models loaded across calls (`FusionMatcher().link(query_texts, corpus_texts)`).

`link` returns one row per query:

| Column | Description |
|---|---|
| `query_idx`, `query_text` | query row and its record text |
| `match_idx`, `match_text` | matched corpus row and its text (missing only for an empty corpus) |
| `score` | the confidence (below) |
| `margin` | top minus runner-up fused score (0 when tied at the top; missing for a single candidate) |
| `reranker_probability` | mean 0-1 reranker score (sigmoid of the logit) of the match, over the rerankers |
| `fused_score` | the match's fused score |

### R

```r
results <- ensemble_link(queries, corpus, return_scores = TRUE)

# several fields: field=value pairs joined by " | ", with multifield = TRUE
q <- paste0("first=", df$first, " | last=", df$last)
r <- paste0("first=", ref$first, " | last=", ref$last)
results <- ensemble_link(q, r, multifield = TRUE)
```

`return_scores = TRUE` adds `match_index`, `score`, `margin` and `reranker_probability`. `ensemble_link_blocked()` does hierarchical matching.

## Confidence

`score` is, for each query, the mean of two percentile ranks among **all queries linked in the same call**: the rank of its fused top-minus-runner-up margin and the rank of its match's mean reranker probability (average ranks for ties):

    score = (rank(margin) + rank(reranker_probability)) / (2 n)

A query whose top fused score is tied between candidates gets 0. The score does not change which record is matched. It orders the matches from least to most trustworthy, so that you can discard likely false links, for example queries with no counterpart in the corpus.

Because the score is a rank within the call, it is relative, not a probability:

- the scores of a call are spread over (0, 1] by construction, however good or bad its matches are, so a threshold removes a share of the weakest links of that call rather than links below a fixed quality;
- a threshold chosen on one set of queries (for example a labeled sample) carries over to another call only when the two query sets are alike in composition and size;
- linking a batch in pieces gives different scores than linking it at once, and the top match can also shift slightly, because the agreement weights and the hubness term use the whole query set. Link the queries that you want to compare in one call.

`reranker_probability` is an absolute score, comparable across calls.

## Speed

**Timings.** On one NVIDIA A100 (80 GB), linking 1,000 five-field voter records against a corpus of one million records takes about 10 minutes, 9.4 of them spent embedding and indexing the corpus. Scoring the candidates takes the same time whatever the size of the corpus, so the time per query grows with the corpus only through retrieval. A GPU is strongly recommended; on CPU the models run in float32 and are much slower.

**Exact and fast mode.** With `exact=True` (the default), the models see fixed batches: embeddings in calls of 10,000 texts at batch size 256, reranker pairs in query order at batch size 128, and dense and TF-IDF scores computed on the GPU 16 queries at a time. With `exact=False`, each distinct query text, corpus text and (query, candidate) pair goes through the models once, in length-sorted batches. Fast mode saves time when records repeat (occupational titles, common names). Its bfloat16 scores differ from exact mode at noise level, which can reorder nearly tied candidates.

**Index cache.** Embedding the corpus is the slowest step for a large corpus. With `index_cache="/some/dir"`, the first call saves the corpus index (embeddings, TF-IDF matrix and vectorizer) and later calls with the same corpus load it:

```python
a = link(batch_a, corpus, column_query="name", index_cache="/path/to/cache")
b = link(batch_b, corpus, column_query="name", index_cache="/path/to/cache")  # corpus not re-embedded
```

The cache key is a SHA-256 over the corpus texts in order, the embedding model and revision, precision, batching mode, TF-IDF settings, library versions and GPU model, so a hit is the index that would have been recomputed. The vectorizer is stored with pickle, so use only a directory that you trust.

## Models, precision and licenses

| Model | Role | Parameters | License |
|---|---|---:|---|
| `microsoft/harrier-oss-v1-0.6b` | embedding | 0.6B | MIT |
| `jinaai/jina-reranker-v2-base-multilingual` | reranker (default) | 278M | CC-BY-NC-4.0 (non-commercial) |
| `BAAI/bge-reranker-v2-m3` | reranker (default) | 568M | Apache-2.0 |
| `zeroentropy/zerank-2-reranker` | reranker (optional) | 4.0B | Apache-2.0 |

The package code is MIT-licensed. Default models are loaded at pinned Hugging Face revisions. On a GPU all models run in bfloat16 (embeddings are normalized and stored in float32, and fusion is in float64); on CPU they run in float32. `dtype="float32"` forces full precision on a GPU.

Because of Jina v2, **the default model set may be used for research and other non-commercial purposes only.** For commercial use, replace Jina v2 with zerank-2, so that every model is MIT or Apache-2.0. zerank-2 has 4 billion parameters and scores pairs about nine times more slowly than the default rerankers.

```python
from zeroshot_linkage import link, COMMERCIAL_RERANKER_MODELS
results = link(queries, corpus, column_query="name", reranker_models=COMMERCIAL_RERANKER_MODELS)
```

```r
results <- ensemble_link(queries, corpus, reranker_model = NULL,
                         reranker_model_3 = "zeroentropy/zerank-2-reranker")
```

Other models can be passed (`embedding_model=`, `reranker_models=[...]` in Python; `embedding_model`, `reranker_model`, `reranker_model_2`, `reranker_model_3` in R): any sentence-transformers embedding model, and rerankers that are sequence-classification cross-encoders with a single relevance logit. A third reranker can be added to the default pair with `reranker_models=DEFAULT_RERANKER_MODELS + (ZERANK_RERANKER,)` (`from zeroshot_linkage.core import ZERANK_RERANKER`).

## Citation

```bibtex
@unpublished{dasanaike2026zeroshot,
  title  = {Pre-Trained Language Models as Zero-Shot Tools for Social Science Research},
  author = {Dasanaike, Noah},
  year   = {2026},
  note   = {Working paper}
}
```
