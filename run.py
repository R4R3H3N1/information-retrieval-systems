import os
import time

import retrieval
import configuration
import retrieval_metrics
import LSI.retrieval_lsi


def query(indexer):
    while True:
        query_string = input("Enter query: ")
        k = input("Enter top k: ")
        if query_string == "exit()":
            break
        print("--------------------------------------------------")
        start = time.time()
        result = indexer.retrieve_k(query_string, int(k))
        print(f'Found {len(result)} documents in {round(time.time() - start, 3)} seconds:')
        print(result)
        print("--------------------------------------------------")


if __name__ == '__main__':

    # Create and Evaluate Vector Space Model
    i_vector_space = retrieval.VectorSpaceModel(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    #query(i_vector_space)
    scorer = retrieval_metrics.RetrievalScorer(i_vector_space)
    scorer.eval()

    # # Create LSI model
    i_lsi = LSI.retrieval_lsi.LatentSemanticIndex(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    #result = i_lsi.retrieve_k("berries to prevent muscle soreness", configuration.TOP_K_DOCS)
    # #print(f"Result: {result}")
    #scorer = retrieval_metrics.RetrievalScorer(i_lsi)
    #scorer.eval()
