import time
from typing import List
import algorithms
import indexer


def execute_query(index: indexer.Index, query: str) -> List[int]:
    print(f"Starting query: {query}")
    start = time.time()
    result = algorithms.fast_cosine_score(index, query)
    print(f"Query found {len(result)} results in {time.time() - start}.")
    return result
