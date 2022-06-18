import os
import time

import retrieval
import configuration
import retrieval_metrics
import LSI.retrieval_lsi

if __name__ == '__main__':

    # Create and Evaluate Vector Space Model
    i_vector_space = retrieval.VectorSpaceModel(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    scorer = retrieval_metrics.RetrievalScorer(i_vector_space)
    scorer.eval()

    # # Create LSI model
    # i_lsi = LSI.retrieval_lsi.LatentSemanticIndex(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    # #result = i_lsi.retrieve_k("berries to prevent muscle soreness", configuration.TOP_K_DOCS)
    # #print(f"Result: {result}")
    # scorer = retrieval_metrics.RetrievalScorer(i_lsi)
    # scorer.eval()
