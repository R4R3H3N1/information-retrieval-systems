import os
import retrieval
import configuration
import retrieval_metrics

if __name__ == '__main__':

    i = retrieval.VectorSpaceModel(os.path.join(os.getcwd(), "data", configuration.DOCS_FILE))
    scorer = retrieval_metrics.RetrievalScorer(i)
    scorer.eval()
