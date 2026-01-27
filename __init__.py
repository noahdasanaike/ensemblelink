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
"""

from .linker import link

__version__ = "1.0.0"
__all__ = ["link"]
