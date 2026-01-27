#!/usr/bin/env python3
"""
Example: Zero-shot record linkage

Run from the zeroshot_linkage directory:
    python example.py
"""

import pandas as pd
from zeroshot_linkage import link

# Sample data: city names with typos/variations
queries = pd.DataFrame({
    "city": [
        "San Fransisco",    # typo
        "Los Angelas",      # typo
        "New York City",    # variation
        "Philly",           # nickname
        "Seattle",          # exact match
    ]
})

corpus = pd.DataFrame({
    "city": [
        "San Francisco",
        "Los Angeles",
        "New York",
        "Philadelphia",
        "Seattle",
        "Chicago",
        "Houston",
    ]
})

print("Queries:")
print(queries)
print("\nCorpus:")
print(corpus)

# Link them
results = link(queries, corpus, column_query="city", threshold=0.3)

print("\nResults:")
print(results.to_string(index=False))
