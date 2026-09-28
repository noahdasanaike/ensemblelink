"""
EnsembleLink: zero-shot record linkage with four-expert agreement fusion.

Candidates are retrieved by exact dense search plus character TF-IDF and scored
by four experts (a Jina v2 + BGE v2-m3 reranker ensemble, a CSLS-corrected dense
cosine, a sparse TF-IDF cosine, and Jaro-Winkler similarity). The experts are
fused without any labeled data by weighting each one by how often it agrees with
the consensus pick. The returned ``score`` is the B2 confidence, which ranks the
queries of one call. All models run locally; no API keys are required.

Example:
    from zeroshot_linkage import link

    results = link(
        queries, corpus,
        column_query="name",
        column_corpus="name",
    )

For hierarchical matching (e.g. states then counties):
    from zeroshot_linkage import link_blocked

    results = link_blocked(
        queries, corpus,
        blocking_query="state", detail_query="county",
    )

For direct control over the matcher (reuse loaded models across calls):
    from zeroshot_linkage import FusionMatcher

    matcher = FusionMatcher()
    result = matcher.link(query_texts, corpus_texts)   # dict of per-query arrays
"""

from .linker import link, link_blocked
from .core import FusionMatcher, DEFAULT_RERANKER_MODELS, COMMERCIAL_RERANKER_MODELS
from .occupations import build_unified_corpus, deduplicate_corpus, EnsembleOcc

__version__ = "2.1.0"
__all__ = [
    "link",
    "link_blocked",
    "FusionMatcher",
    "DEFAULT_RERANKER_MODELS",
    "COMMERCIAL_RERANKER_MODELS",
    "build_unified_corpus",
    "deduplicate_corpus",
    "EnsembleOcc",
]
