"""
EnsembleLink: zero-shot record linkage with four-expert agreement fusion.

Candidates are retrieved by an embedding-plus-TF-IDF ensemble and scored by four
experts (a two-model reranker ensemble, a CSLS-corrected dense cosine, a sparse
TF-IDF cosine, and Jaro-Winkler similarity). The experts are fused without any
labeled data by weighting each one by how often it agrees with the consensus
pick. All models run locally; no API keys are required.

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
    pairs = matcher.match(query_texts, corpus_texts)
"""

from .linker import link, link_blocked
from .core import FusionMatcher
from .occupations import build_unified_corpus, deduplicate_corpus, EnsembleOcc

__version__ = "2.0.0"
__all__ = [
    "link",
    "link_blocked",
    "FusionMatcher",
    "build_unified_corpus",
    "deduplicate_corpus",
    "EnsembleOcc",
]
