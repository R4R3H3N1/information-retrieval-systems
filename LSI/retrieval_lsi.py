from __future__ import annotations

import time
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
        print("Started creating index.")
        start = time.time()
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
        print(f"Finished creating index in {round(time.time() - start, 2)} seconds.")

    # --------------------------------------------------------------------------- #
    def fill_term_doc_matrix(self):
        print("Started creating term document matrix.")
        start = time.time()
        # Fill term document matrix
        term_index_mapping_list = []
        term_index = 0
        for term, doc_ids in self.dictionary.items():
            term_index_mapping_list.append(term)
            for doc_id in doc_ids:
                self.term_doc_matrix[term_index, np.argwhere(self.doc_id_index_mapping == doc_id)] += 1

            term_index += 1

        self.term_index_mapping = np.array(term_index_mapping_list)
        print(f"Finished creating term document matrix in {round(time.time() - start, 2)} seconds.")

    # --------------------------------------------------------------------------- #
    def calculate_latent_space(self):
        d = configuration.LSI_D

        print(f"Started calculating latent space with {d} dimensions.")
        start = time.time()

        u, s, v = svd(self.term_doc_matrix, full_matrices=False)
        ud = u[:, 0:d]
        sd = s[0:d]
        vd = v[0:d, :]
        sd_sqrt = np.diag(np.sqrt(sd))
        self.ud_prime = np.matmul(ud, sd_sqrt)
        self.vd_prime = np.matmul(sd_sqrt, vd)
        self.sd_inv = np.linalg.inv(np.diag(sd))

        print(f"Finished calculating latent space in {round(time.time() - start, 2)} seconds.")

    # --------------------------------------------------------------------------- #
    def retrieve(self, query):
        query_vector = self.map_query_to_vector(query)
        latent_space_query_vec = self.map_query_vec_to_latent_space(query_vector)

        query_doc_sim = np.zeros((len(self.doc_id_index_mapping)))

        for doc in range(len(self.doc_id_index_mapping)):
            doc_vec = self.vd_prime[:, doc]
            query_doc_sim[doc] = cosinus_similarity(latent_space_query_vec, np.transpose(doc_vec))

        query_doc_sim[np.isnan(query_doc_sim)] = 0

        return query_doc_sim

    # --------------------------------------------------------------------------- #
    def retrieve_k(self, query, k):
        if configuration.LOGGING:
            print(f"Started executing query: {query}")
        start = time.time()
        result = self.get_top_k(self.retrieve(query), k)
        if configuration.LOGGING:
            print(f"Finished executing query in {round(time.time() - start, 2)} seconds.")
        return result

    # --------------------------------------------------------------------------- #
    def map_query_to_vector(self, query):
        query_terms = tokenizer.create_token_stream(query)

        query_vector = np.zeros((len(self.dictionary.keys())))

        for index, term in enumerate(self.term_index_mapping):
            if term in query_terms:
                query_vector[index] = 1

        return query_vector

    # --------------------------------------------------------------------------- #
    def map_query_vec_to_latent_space(self, vec):
        return np.matmul(np.matmul(self.sd_inv, np.transpose(self.ud_prime)), vec)

    def get_top_k(self, query_doc_sims, k):
        top_k_docids = []

        for i in range(k):
            doc_index = np.argmax(query_doc_sims)
            top_k_docids.append((int(self.doc_id_index_mapping[doc_index]), query_doc_sims[doc_index]))
            query_doc_sims[doc_index] = 0
        return top_k_docids

