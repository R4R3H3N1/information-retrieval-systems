from typing import List

import numpy as np

import configuration
import indexer
import tokenizer


def calc_query_terms_dictionary(terms: List[str]) -> dict:
    terms_dic = {}
    for term in terms:
        try:
            terms_dic[term] += 1
        except KeyError:
            terms_dic[term] = 1

    return terms_dic


# TODO implement method to return k doc ids with highest score
def get_top_k_docs(scores: np.ndarray, k: int) -> List[int]:
    top_k_doc_ids = []
    for i in range(k):
        new_result_doc_id = np.argmax(scores)
        top_k_doc_ids.append(new_result_doc_id)
        scores[new_result_doc_id] = 0
    return top_k_doc_ids


def calc_w_f(index: indexer.Index, term: str, query_freq: int, doc_id: int, posting_list_obj: indexer.Postinglist) -> float:
    term_doc_freq = len(posting_list_obj.positions[doc_id])
    len_doc = index.documentIDs[doc_id]
    n = len(index.dictionary.items())
    d_f_t = posting_list_obj.occurrence
    return query_freq * (term_doc_freq / (term_doc_freq + (configuration.K * (len_doc / index.average_doc_len)))
                         * np.log((n / d_f_t)))


def fast_cosine_score(index: indexer.Index, query: str) -> List[int]:
    query_terms = tokenizer.get_token_from_line(query)
    query_terms_dic = calc_query_terms_dictionary(query_terms)
    # TODO maybe use dict as doc ids are not continuous thus lot of unused array space
    scores = np.zeros((np.max(list(index.documentIDs.keys())) + 1))
    for query_term, query_term_freq in query_terms_dic.items():
        posting_list_obj = index.dictionary[index.termClassMapping[query_term]]
        for doc_id in posting_list_obj.plist:
            scores[doc_id] += calc_w_f(index, query_term, query_term_freq, doc_id, posting_list_obj)
    for doc_id, len_doc in index.documentIDs.items():
        scores[doc_id] = scores[doc_id] / len_doc
    return get_top_k_docs(scores, configuration.TOP_K_DOCS)
