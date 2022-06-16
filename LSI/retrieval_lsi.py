from __future__ import annotations

from collections import Counter
from scipy.linalg import svd
import numpy as np

import configuration
import retrieval
import tokenizer


# --------------------------------------------------------------------------- #
def cosinus_similarity(vec1, vec2):
    try:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except ZeroDivisionError:
        return 0.0


class LatentSemanticIndex(retrieval.InitRetrievalSystem):
    def __init__(self, filename: str):
        self.dictionary = {}
        self.doc_id_index_mapping = []
        self.term_index_mapping = []
        self.term_doc_matrix = None

        # Latent Space
        self.sd_inv = None
        self.ud_prime = None
        self.vd_prime = None

        self.invoke_toknizer(filename)
        self.fill_term_doc_matrix()
        self.calculate_latent_space()

    # --------------------------------------------------------------------------- #
    def invoke_toknizer(self, filename: str) -> None:
        try:
            with open(filename, 'r', encoding='utf8') as f:
                file = f.read()
                docs = file.split('\n')
        except FileNotFoundError:
            print(f'ERROR: file {filename} not found')
            return

        self.doc_id_index_mapping = np.zeros((len(docs)))
        doc_id_index = 0
        for docid, tokens in tokenizer.tokenize_documents(docs):
            # Used for Term Doc Matrix
            self.doc_id_index_mapping[doc_id_index] = docid
            doc_id_index += 1

            for token in tokens:
                try:
                    self.dictionary[token].append(docid)
                except KeyError:
                    self.dictionary[token] = [docid]

        # Init term document matrix
        self.term_doc_matrix = np.zeros([len(self.dictionary.keys()), doc_id_index + 1])

    # --------------------------------------------------------------------------- #
    def fill_term_doc_matrix(self):
        # Fill term document matrix
        term_index_mapping_list = []
        term_index = 0
        for term, doc_ids in self.dictionary.items():
            for doc_id in doc_ids:
                self.term_doc_matrix[term_index, np.argwhere(self.doc_id_index_mapping == doc_id)] += 1
                term_index_mapping_list.append(term)

            term_index += 1

        self.term_index_mapping = np.array(term_index_mapping_list)

    # --------------------------------------------------------------------------- #
    def calculate_latent_space(self):
        d = configuration.LSI_D
        u, s, v = svd(self.term_doc_matrix, full_matrices=False)
        ud = u[:, 0:d]
        sd = s[0:d]
        vd = v[0:d, :]
        sd_sqrt = np.diag(np.sqrt(sd))
        self.ud_prime = np.matmul(ud, sd_sqrt)
        self.vd_prime = np.matmul(sd_sqrt, vd)
        self.sd_inv = np.linalg.inv(np.diag(sd))

    # --------------------------------------------------------------------------- #
    def retrieve(self, query):
        query_terms = tokenizer.create_token_stream(query)
        query_tf = Counter(query_terms)

        # TODO implement retrieve with cosinus similarity

        return []

    # --------------------------------------------------------------------------- #
    def retrieve_k(self, query, k):

        return retrieval.get_top_k(self.retrieve(query), k)

