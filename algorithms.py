from typing import List

import numpy as np

import configuration
import indexer
import tokenizer


def get_top_k_docs(scores: np.ndarray, k: int) -> List[int]:
    print(str(k))
    return [scores[0]]


def calc_w_f(term: str, doc: int) -> int:
    return 1


def fast_cosine_score(index: indexer.Index, query: str) -> List[int]:
    query_terms = tokenizer.get_token_from_line(query)
    scores = np.zeros((len(index.documentIDs)))
    for query_term in query_terms:
        posting_list_obj = index.dictionary[index.termClassMapping[query_term]]
        for doc_id in posting_list_obj.plist:
            scores[doc_id] += calc_w_f(query_term, doc_id)
    for doc_id, len_doc in index.documentIDs:
        scores[doc_id] = scores[doc_id] / len_doc
    return get_top_k_docs(scores, configuration.K)
