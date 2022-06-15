from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tokenizer import tokenize_documents
import tokenizer
# TODO clean imports (favor explicit: from ... import ...)

from collections import Counter
import json
import time
import os
from typing import Type, List, Set

import configuration


# =========================================================================== #
class InitRetrievalSystem(ABC):
    @abstractmethod
    def __init__(self, docIds):
        self.docIds = docIds
        pass

    @abstractmethod
    def retrieve(self, query):
        pass

    @abstractmethod
    def retrieve_k(self, query, k):
        pass

# =========================================================================== #
class VectorSpaceModel(InitRetrievalSystem):
    def __init__(self, filename: str):
        self.dictionary = {}
        self.term_index_mapping = {}
        self.docid_length_mapping = {}

        self.invoke_toknizer(filename
                             )

    # --------------------------------------------------------------------------- #
    def invoke_toknizer(self, filename: str) -> None:
        try:
            with open(filename, 'r', encoding='utf8') as f:
                file = f.read()
                docs = file.split('\n')
        except FileNotFoundError:
            print(f'ERROR: file {filename} not found')
            return

        for docid, tokens in tokenize_documents(docs):
            self.docid_length_mapping[docid] = len(tokens)
            position_count = 1
            for token in tokens:
                try:
                    ti = self.term_index_mapping[token]
                except KeyError:
                    ti = TermIndex(token)
                    self.term_index_mapping[token] = ti

                try:
                    self.dictionary[ti].append(docid, position_count)
                    ti.occurence += 1
                except KeyError:
                    self.dictionary[ti] = Postinglist(docid, position_count)

                position_count += 1

        for key, val in self.dictionary.items():
            val.final_sort_postinglist()

    # --------------------------------------------------------------------------- #
    def retrieve(self, query):
        query_terms = tokenizer.create_token_stream(query)
        query_tf = Counter(query_terms)

        # TODO sparse matrix, docids not continuous
        # TODO where to add log10?
        MAX_DOCID = np.max(list(self.docid_length_mapping.keys()))
        N_DOCUMENTS = len(self.docid_length_mapping)
        scores = np.zeros((MAX_DOCID + 1))  # TODO

        for q_term, q_term_tf in query_tf.items():
            try:
                postinglist_obj = self.dictionary[self.term_index_mapping[q_term]]
            except KeyError as k:
                #print(f'term {q_term} not present in corpus')
                continue

            for docid in postinglist_obj.plist:
                scores[docid] += self.calc_score(N_DOCUMENTS, postinglist_obj, docid)

        # TODO vectorizable
        for docid, _len in self.docid_length_mapping.items():
            scores[docid] = scores[docid] / _len

        return np.argwhere(scores != 0).flatten().tolist()

    # --------------------------------------------------------------------------- #
    def calc_score(self, n_docs: int, postinglist_obj: Postinglist, docid: int) -> float:
        tf = len(postinglist_obj.positions[docid])
        idf = n_docs / len(postinglist_obj.plist)
        return (1 + np.log10(tf)) * np.log10(idf)

    # --------------------------------------------------------------------------- #
    def retrieve_k(self, query, k):

        return self.get_top_k(np.asarray(self.retrieve(query)), k)

    # --------------------------------------------------------------------------- #
    def get_top_k(self, scores, k):
        top_k_docids = []
        if scores.size == 0:
            return []
        
        for i in range(k):
            docid = np.argmax(scores)
            top_k_docids.append((docid, scores[docid]))
            scores[docid] = 0
        return top_k_docids

# =========================================================================== #
class TermIndex:
    __slots__ = ('term', 'occurence')

    def __init__(self, term: str):
        self.term = term
        self.occurence = 1

    def __hash__(self):
        return hash(self.term)


# =========================================================================== #
class Postinglist:
    __slots__ = ('plist', 'seen_docids', 'occurrence', 'positions')

    def __init__(self, docid: int = None, position: int = None):
        self.plist = []
        self.occurrence = 0
        self.seen_docids = set()
        self.positions = {}

        if docid:
            self.append(docid, position)

    def __len__(self):
        return len(self.plist)

    def __getitem__(self, idx):
        return self.plist[idx]

    # --------------------------------------------------------------------------- #
    def append(self, docid: int, position: int) -> None:
        try:
            self.positions[docid].append(position)
            self.occurrence += 1
        except KeyError:
            self.positions[docid] = [position]
            self.occurrence += 1

        if docid in self.seen_docids:
            pass
        else:
            self.plist.append(docid)
            self.seen_docids.add(docid)

    # --------------------------------------------------------------------------- #
    def final_sort_postinglist(self) -> None:
        self.plist = sorted(self.plist)