"""
Zero-Shot Record Linkage with Ensemble Retrieval and Cross-Encoder Reranking

A simple, API-key-free library for linking records across datasets using
pre-trained language models.

Example:
    from zeroshot_linkage import link

    results = link(
        queries, corpus,
        column_query="name",
        column_corpus="name"
    )

For hierarchical matching (e.g., states then counties):
    from zeroshot_linkage import link_blocked

    results = link_blocked(
        queries, corpus,
        blocking_query="state", detail_query="county"
    )
"""

from .linker import link, link_blocked

__version__ = "1.0.0"
__all__ = ["link", "link_blocked"]
