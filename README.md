# EnsembleLink

EnsembleLink performs record linkage with pre-trained language models and requires no training data, labeled examples, or API keys. Bug reports are welcome.

EnsembleLink is described in Noah Dasanaike, [Pre-Trained Language Models as Zero-Shot Tools for Social Science Research](https://www.dropbox.com/scl/fi/2kx17wvydhr3v8m4u59ek/zeroshot_llms_dasanaike.pdf?rlkey=kybmjjnptca1urqcin06ro349&st=fk6w43uj&e=1&dl=0).

## How it works

For each query, EnsembleLink retrieves a pool of candidate records and then scores each candidate with four experts.

**Retrieval.** The pool is the union of the 30 corpus records that are closest to the query by embedding cosine similarity (an exact search over the full corpus using `microsoft/harrier-oss-v1-0.6b`) and the 30 records that are closest by TF-IDF over character 2-4-grams of the transliterated text. For a query that shares no character n-gram with any record, such as a name written in another script, the pool instead consists of the 60 records that are closest by embedding.

**Experts.**

1. **Reranker ensemble.** Two cross-encoders, Jina Reranker v2 (multilingual) and BGE Reranker v2-m3, score the query and the candidate jointly as a single input. The expert score is the sum of the z-scores of the two rerankers within the pool.
2. **CSLS-corrected dense cosine.** The expert score is `2 * cosine - hubness`, where the hubness of a record is the mean cosine similarity between that record and the 10 queries nearest to it. This correction discounts records that are close to every query.
3. **Sparse TF-IDF cosine** on text transliterated with Unidecode. This expert allows records that differ by abbreviations and typos to be matched.
4. **Jaro-Winkler similarity** on the transliterated text.

**Fusion without labels.** The scores of each expert are z-scored within the pool. Each expert then votes for its top candidate, and an expert that assigns the same score to each candidate in a pool abstains. The weight of each expert is the squared share of queries for which that expert agrees with the consensus vote. The match is the candidate with the highest weighted sum.

**Record text.** A record with several fields is written as `field=value | field=value`, with missing fields omitted. In this case, each query is embedded with the instruction "Retrieve the record that refers to the same entity". A record with a single field is written as the value alone, and queries are embedded without an instruction.

## Installation

### Python

```bash
pip install git+https://github.com/noahdasanaike/ensemblelink.git
```

If you have a GPU, first install a CUDA build of PyTorch (https://pytorch.org/get-started/locally/). On the first run, the package downloads the embedding model and the two rerankers (about 4 GB in total).

### R

```r
devtools::install_github("noahdasanaike/ensemblelink/r_package")
library(ensemblelink)
install_ensemblelink()   # installs the Python dependencies, once; then restart R
```

The R package calls the code of the Python package, using a vendored copy in `inst/python`. The two packages therefore produce the same results. To use a specific Python installation, call `configure_python(condaenv = "myenv")` or `configure_python(python = "/path/to/python")`. The full R interface is documented in [`r_package/README.md`](r_package/README.md).

## Usage

### Python

```python
import pandas as pd
from zeroshot_linkage import link

queries = pd.DataFrame({"name": ["John Smith", "Jane Doe", "Robert Johnson"]})
corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson", "R. Johnson"]})

results = link(queries, corpus, column_query="name")
```

- **Several fields:** `link(queries, corpus, columns_query=["first", "last", "birth_year"])`. By default, the corpus columns have the same names as the query columns. Otherwise, pass `columns_corpus`, which is paired with `columns_query` by position. Store whole numbers as integers (pandas `Int64`) in order that they appear in the record text as `1980` instead of `1980.0`.
- **Ties:** pass `id_query` and `id_corpus` (columns of unique IDs) to break exact ties by record ID. Otherwise, ties are broken by row position.
- **Hierarchical matching:** `link_blocked(queries, corpus, blocking_query="state", detail_query="county")` matches on the coarse field first and then on the detail field within the matched block.
- **Repeated calls:** `FusionMatcher` retains the loaded models across calls (`FusionMatcher().link(query_texts, corpus_texts)`).

`link` returns one row per query:

| Column | Description |
|---|---|
| `query_idx`, `query_text` | the query row and its record text |
| `match_idx`, `match_text` | the matched corpus row and its record text (missing only for an empty corpus) |
| `score` | the confidence score (see [Confidence](#confidence)) |
| `margin` | the top fused score minus the runner-up fused score (0 when the top candidates are tied; missing when the pool has a single candidate) |
| `reranker_probability` | the 0-1 reranker score of the match (the sigmoid of the logit), averaged over the rerankers |
| `fused_score` | the fused score of the match |

### R

```r
results <- ensemble_link(queries, corpus, return_scores = TRUE)

# several fields: field=value pairs joined by " | ", with multifield = TRUE
q <- paste0("first=", df$first, " | last=", df$last)
r <- paste0("first=", ref$first, " | last=", ref$last)
results <- ensemble_link(q, r, multifield = TRUE)
```

With `return_scores = TRUE`, the output also includes `match_index`, `score`, `margin`, and `reranker_probability`. `ensemble_link_blocked()` performs hierarchical matching.

## Confidence

For each query, `score` is the mean of two percentile ranks, each computed among **all queries linked in the same call**. The first is the rank of the margin between the top and runner-up fused scores of the query. The second is the rank of the mean reranker probability of the matched record. Tied values receive average ranks:

    score = (rank(margin) + rank(reranker_probability)) / (2 n)

A query for which two or more candidates share the top fused score receives a score of 0. The score has no effect on which record is matched. The score instead orders the matches from least to most trustworthy, which allows you to discard likely false links, for instance queries that have no counterpart in the corpus.

Because the score is a rank within a single call, it is a relative measure and not a probability. In particular:

- The scores of a given call are spread over (0, 1] by construction, regardless of the quality of its matches. A threshold therefore removes a share of the weakest links in that call, and does not remove the links below a fixed level of quality.
- A threshold chosen on one set of queries (for instance, a labeled sample) applies to another call only when the two sets of queries are similar in composition and size.
- Linking a batch in several pieces produces different scores than linking the full batch in a single call, and the top match can also shift slightly. Both differences arise because the agreement weights and the hubness term are computed over the full set of queries in the call. Link all of the queries that you want to compare in a single call.

By contrast, `reranker_probability` is an absolute score and can be compared across calls.

## Speed

**Timings.** On a single NVIDIA A100 (80 GB), linking 1,000 five-field voter records to a corpus of one million records takes about 10 minutes, of which 9.4 minutes are spent embedding and indexing the corpus. Because scoring the candidates takes the same time regardless of the size of the corpus, the time per query increases with the size of the corpus only through retrieval. A GPU is strongly recommended. On a CPU, the models are run in float32 and are considerably slower.

**Exact and fast mode.** With `exact=True` (the default), the inputs are passed to the models in fixed batches. Embeddings are computed in calls of 10,000 texts at batch size 256, reranker pairs are scored in query order at batch size 128, and dense and TF-IDF scores are computed on the GPU 16 queries at a time. With `exact=False`, each distinct query text, corpus text, and (query, candidate) pair is passed through the models once, in batches sorted by length. Fast mode reduces run time when records repeat, as with occupational titles or common names. The bfloat16 scores of fast mode differ from those of exact mode at the level of noise, and these differences can reorder candidates that are nearly tied.

**Index cache.** For a large corpus, embedding the corpus is the slowest step. With `index_cache="/some/dir"`, the first call saves the index of the corpus (the embeddings, the TF-IDF matrix, and the vectorizer), and later calls with the same corpus load the saved index:

```python
a = link(batch_a, corpus, column_query="name", index_cache="/path/to/cache")
b = link(batch_b, corpus, column_query="name", index_cache="/path/to/cache")  # corpus not re-embedded
```

The cache key is a SHA-256 hash of the corpus texts in order, the embedding model and its revision, the precision, the batching mode, the TF-IDF settings, the library versions, and the GPU model. A cache hit is therefore the same index that would otherwise have been recomputed. Because the vectorizer is stored with pickle, use only a cache directory that you trust.

## Models, precision, and licenses

| Model | Role | Parameters | License |
|---|---|---:|---|
| `microsoft/harrier-oss-v1-0.6b` | embedding | 0.6B | MIT |
| `jinaai/jina-reranker-v2-base-multilingual` | reranker (default) | 278M | CC-BY-NC-4.0 (non-commercial) |
| `BAAI/bge-reranker-v2-m3` | reranker (default) | 568M | Apache-2.0 |
| `zeroentropy/zerank-2-reranker` | reranker (optional) | 4.0B | Apache-2.0 |

The package code is released under the MIT license. The default models are loaded at pinned Hugging Face revisions. On a GPU, all models are run in bfloat16. The embeddings are then normalized and stored in float32, and the fusion step is computed in float64. On a CPU, the models are run in float32. To force full precision on a GPU, set `dtype="float32"`.

Because Jina v2 is licensed for non-commercial use only, **the default model set may be used only for research and other non-commercial purposes.** For commercial use, replace Jina v2 with zerank-2, in which case each model is licensed under MIT or Apache-2.0. However, zerank-2 has 4 billion parameters and scores pairs about nine times more slowly than the default rerankers.

```python
from zeroshot_linkage import link, COMMERCIAL_RERANKER_MODELS
results = link(queries, corpus, column_query="name", reranker_models=COMMERCIAL_RERANKER_MODELS)
```

```r
results <- ensemble_link(queries, corpus, reranker_model = NULL,
                         reranker_model_3 = "zeroentropy/zerank-2-reranker")
```

Other models can be passed using `embedding_model=` and `reranker_models=[...]` in Python, or `embedding_model`, `reranker_model`, `reranker_model_2`, and `reranker_model_3` in R. The embedding model can be any sentence-transformers embedding model, and each reranker must be a sequence-classification cross-encoder with a single relevance logit. To add a third reranker to the default pair, pass `reranker_models=DEFAULT_RERANKER_MODELS + (ZERANK_RERANKER,)` after importing `ZERANK_RERANKER` from `zeroshot_linkage.core`.

## Citation

```bibtex
@unpublished{dasanaike2026zeroshot,
  title  = {Pre-Trained Language Models as Zero-Shot Tools for Social Science Research},
  author = {Dasanaike, Noah},
  year   = {2026},
  note   = {Working paper}
}
```
