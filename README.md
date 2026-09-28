# EnsembleLink

Accurate record linkage without training data, using pre-trained language models. No labeled examples or API keys required. Bug reports welcome.

**Paper**: Noah Dasanaike, [Pre-Trained Language Models as Zero-Shot Tools for Social Science Research](https://www.dropbox.com/scl/fi/2kx17wvydhr3v8m4u59ek/zeroshot_llms_dasanaike.pdf?rlkey=kybmjjnptca1urqcin06ro349&st=fk6w43uj&e=1&dl=0). The package's defaults are the specification evaluated in the paper (2026-09-27), and with them it reproduces the paper's EnsembleLink results (see [Reproducing the paper](#reproducing-the-paper)).

## How it works

For each query, EnsembleLink retrieves a candidate pool: the union of the 30 corpus records closest by embedding cosine (exact search over the whole corpus with `microsoft/harrier-oss-v1-0.6b`) and the 30 closest by character 2-4-gram TF-IDF on transliterated text. A query that shares no character n-gram with any record (for example a name in another script) gets the 60 closest by embedding instead. Every candidate is scored by four experts:

1. **Reranker ensemble**: two cross-encoders, Jina Reranker v2 (multilingual) and BGE Reranker v2-m3, read the query and candidate together; the expert is the sum of their per-pool z-scores.
2. **CSLS-corrected dense cosine**: `2 * cosine - hubness`, where a record's hubness is its mean cosine with its 10 nearest queries, so records close to everything are discounted.
3. **Sparse TF-IDF cosine** on transliterated text (Unidecode), which catches abbreviations and typos.
4. **Jaro-Winkler** similarity on transliterated text.

Each expert is z-scored within the pool, and the experts are fused **without labeled data**: each expert votes for its top candidate, an expert with no spread over a pool abstains, and each expert's weight is the squared share of queries on which it agrees with the consensus vote. The match is the candidate with the highest weighted sum.

Record text follows the field-count rule: a record with several fields is written as `field=value | field=value` (missing fields omitted), and its query gets the embedding instruction "Retrieve the record that refers to the same entity"; a single field is written as its value alone, with no instruction.

## Python

```bash
pip install git+https://github.com/noahdasanaike/ensemblelink.git
```

Install a CUDA build of PyTorch first if you have a GPU (https://pytorch.org/get-started/locally/). The first run downloads the embedding model and the two rerankers (about 4 GB).

```python
import pandas as pd
from zeroshot_linkage import link

queries = pd.DataFrame({"name": ["John Smith", "Jane Doe", "Robert Johnson"]})
corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson", "R. Johnson"]})

results = link(queries, corpus, column_query="name")
```

Several fields: `link(queries, corpus, columns_query=["first", "last", "birth_year"])` (corpus columns default to the same names, or pass `columns_corpus`, paired by position). Store whole numbers as integers (pandas `Int64`) so that they read `1980`, not `1980.0`. Pass `id_query` and `id_corpus` (columns of unique IDs) to break exact ties by record ID, as the paper does; otherwise row positions are used.

`link` returns one row per query:

| Column | Description |
|---|---|
| `query_idx`, `query_text` | query row and its record text |
| `match_idx`, `match_text` | matched corpus row and its text (missing only for an empty corpus) |
| `score` | the confidence (below) |
| `margin` | top minus runner-up fused score (0 when tied at the top; missing for a single candidate) |
| `reranker_probability` | mean 0-1 reranker score (sigmoid of the logit) of the match, over the rerankers |
| `fused_score` | the match's fused score |

`link_blocked(queries, corpus, blocking_query="state", detail_query="county")` matches a coarse field first and then the detail within the matched block. `FusionMatcher` keeps the models loaded across calls (`FusionMatcher().link(query_texts, corpus_texts)`).

## R

```r
devtools::install_github("noahdasanaike/ensemblelink/r_package")
library(ensemblelink)
install_ensemblelink()   # Python dependencies, once; then restart R

results <- ensemble_link(queries, corpus, return_scores = TRUE)
# several fields: field=value pairs joined by " | ", and multifield = TRUE
q <- paste0("first=", df$first, " | last=", df$last)
```

The R package runs the Python package's code (a vendored copy in `inst/python`), so results are the same. `return_scores = TRUE` adds `match_index`, `score`, `margin` and `reranker_probability`. `ensemble_link_blocked()` does hierarchical matching.

## Confidence

`score` is the confidence rule of the paper (B2): for each query, the mean of two percentile ranks among **all queries linked in the same call**, the rank of its fused top-minus-runner-up margin and the rank of its match's mean reranker probability (average ranks for ties):

    score = (rank(margin) + rank(reranker_probability)) / (2 n)

A query whose top fused score is tied between candidates gets 0. The score does not change which record is matched; it orders the matches from least to most trustworthy, for discarding likely false links (for example queries with no counterpart in the corpus).

Because it is a rank within the call, the score is relative, not a probability:

- the scores of a call are spread over (0, 1] by construction, however good or bad its matches are; a threshold removes a share of the weakest links of that call, not links below a fixed quality;
- a threshold chosen on one set of queries (for example on a labeled sample) carries over to another call only when the two query sets are alike in composition and size. The paper selects thresholds on development data and applies them to test queries of the same kind;
- linking a batch in pieces gives different scores than linking it at once (the top-1 can also shift slightly, because the agreement weights and the hubness term use the whole query set). Link the queries you want to compare in one call.

`margin` and `reranker_probability` are returned as well; the reranker probability is an absolute score, comparable across calls.

## Speed

**Exact and fast mode.** `exact=True` (default) feeds the models exactly as the paper's benchmark does: embeddings in calls of 10,000 texts at batch size 256, reranker pairs in query order at batch size 128, dense and TF-IDF scores on the GPU 16 queries at a time. `exact=False` sends each distinct query text, corpus text and (query, candidate) pair through the models once, in length-sorted batches. It is faster, above all when records repeat (occupational titles, common names); bfloat16 scores then move at noise level, which can reorder nearly tied candidates.

**Index cache.** Embedding the corpus is the slowest step for a large corpus. With `index_cache="/some/dir"`, the first call saves the corpus index (embeddings, TF-IDF matrix, vectorizer) and later calls with the same corpus load it:

```python
a = link(batch_a, corpus, column_query="name", index_cache="/path/to/cache")
b = link(batch_b, corpus, column_query="name", index_cache="/path/to/cache")  # corpus not re-embedded
```

The key is a SHA-256 over the corpus texts in order, the embedding model and revision, precision, batching mode, TF-IDF settings, library versions and GPU model, so a hit is the index that would have been recomputed. The vectorizer is stored with pickle: use only a directory you trust.

**Timings.** On one A100 (80 GB), linking 1,000 five-field voter records against a corpus of one million takes about 10 minutes with `exact=True` (9.4 of them embedding and indexing the corpus) and about the same with `exact=False`, which saves time only when texts or pairs repeat (these records are distinct); see the paper's scaling appendix for the full grid. A GPU is strongly recommended; on CPU the models run in float32.

## Models, precision and licenses

| Model | Role | Parameters | License |
|---|---|---:|---|
| `microsoft/harrier-oss-v1-0.6b` | embedding | 0.6B | MIT |
| `jinaai/jina-reranker-v2-base-multilingual` | reranker (default) | 278M | CC-BY-NC-4.0 (non-commercial) |
| `BAAI/bge-reranker-v2-m3` | reranker (default) | 568M | Apache-2.0 |
| `zeroentropy/zerank-2-reranker` | reranker (optional) | 4.0B | Apache-2.0 |

The package code is MIT-licensed. Default models are loaded at the Hugging Face revisions used in the paper. On a GPU all models run in bfloat16 (embeddings are normalized and stored in float32; fusion is float64), as in the paper; on CPU they run in float32. `dtype="float32"` forces full precision on a GPU.

Because of Jina v2, **the default model set may be used for research and other non-commercial purposes only.** For commercial use, replace Jina v2 with zerank-2, so that every model is MIT or Apache-2.0:

```python
from zeroshot_linkage import link, COMMERCIAL_RERANKER_MODELS
results = link(queries, corpus, column_query="name", reranker_models=COMMERCIAL_RERANKER_MODELS)
```

```r
results <- ensemble_link(queries, corpus, reranker_model = NULL,
                         reranker_model_3 = "zeroentropy/zerank-2-reranker")
```

No configuration without Jina v2 met the paper's development rule (no development benchmark worse by more than 0.002 F1). BGE v2-m3 + zerank-2 raised mean development F1 by 0.007 over the default reranker pair across 13 benchmarks but lost 0.010 on North Carolina voters with unmatched queries and 0.008 top-1 on occupations; BGE v2-m3 with gte-multilingual or mxbai-rerank-v2 lost 1 to 2 points on names and products. zerank-2 is also slow: 4B parameters, about nine times Jina v2 + BGE per pair. Adding zerank-2 as a third reranker to the default (`reranker_models=DEFAULT_RERANKER_MODELS + (ZERANK_RERANKER,)`) raised mean development F1 by 0.011 but lowered one benchmark by just over 0.002, and was not adopted.

Other models can be passed (`embedding_model=`, `reranker_models=[...]`): any sentence-transformers embedding model, and rerankers that are sequence-classification cross-encoders with a single relevance logit.

## Reproducing the paper

The replication archive (`benchmark/scaling/runs_final/verify_package_vs_benchmark.py`) links development queries with this package (`exact=True`, record IDs passed for tie-breaks, the benchmark's batch sizes) and compares every stage with the benchmark's own outputs on the same GPU model:

- occupation coding and the Products full join (the reference file is the benchmark's whole corpus): the same top-1 for every query; the same candidate pools, embeddings, hubness and reranker scores bit for bit; the confidence identical for all Products queries and 90% of occupation queries (largest difference 0.005);
- five ParaNames and GeoNames tasks and North Carolina voters: the benchmark embedded the union of the two conditions' reference files and scored both conditions' candidate pairs together, and a bfloat16 embedding or pair score depends slightly on which other records share its batch. Given those batches, the package reproduces the benchmark's top-1 for all but 2 of 8,812 queries (both choices between records with identical text), with identical pools, embeddings, hubness and reranker scores. Linked on one condition's file alone, as a user would, North Carolina keeps the same top-1 for 2,997 of 3,000 queries, and the name tasks change the top-1 of 205 of 2,500 queries, 196 of them choices between records with identical text; development top-1 accuracy moves by -1.6 to +1.3 points and best F1 by -0.7 to +0.8.

The remaining differences come from cuSPARSE, which computes the TF-IDF scores on the GPU as in the benchmark and whose sums can differ in the last bit between runs; they occasionally reorder queries with nearly equal confidence ranks.

## Citation

```bibtex
@unpublished{dasanaike2026zeroshot,
  title  = {Pre-Trained Language Models as Zero-Shot Tools for Social Science Research},
  author = {Dasanaike, Noah},
  year   = {2026},
  note   = {Working paper}
}
```
