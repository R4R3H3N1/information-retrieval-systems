import os
import time

import retrieval
import configuration
import retrieval_metrics
import LSI.retrieval_lsi


def query(indexer):
    while True:
        query_string = input("Enter query: ")
        if query_string == "exit()":
            break
        k = input("Enter top k: ")
        if k == "exit()":
            break
        print("--------------------------------------------------")
        start = time.time()
        result = indexer.retrieve_k(query_string, int(k))
        print(f'Found {len(result)} documents in {round(time.time() - start, 3)} seconds:')
        print(result)
        print("--------------------------------------------------")


if __name__ == '__main__':

    if configuration.MODEL == "VEC":
        # Create and Evaluate Vector Space Model
        i_vector_space = retrieval.VectorSpaceModel(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
        query(i_vector_space)
        #scorer = retrieval_metrics.RetrievalScorer(i_vector_space)
        #scorer.eval()
    elif configuration.MODEL == "LSI":
        # Create LSI model
        i_lsi = LSI.retrieval_lsi.LatentSemanticIndex(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
        query(i_lsi)
        #scorer = retrieval_metrics.RetrievalScorer(i_lsi)
        #scorer.eval()
        