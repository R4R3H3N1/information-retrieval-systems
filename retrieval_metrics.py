import matplotlib.pyplot as plt
import numpy as np

import configuration
import os


# --------------------------------------------------------------------------- #
def precision(y_true, y_pred):
    """
        Calculates precision score for a list of relevant documents and the groundtruth.

        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.

        Returns
        -------
        Score: float
            Precision = TP / (TP + FP)
    """

    y_true_set = set(y_true)
    y_pred_set = set(y_pred)

    tp = y_pred_set.intersection(y_true_set)
    fp = y_pred_set.difference(y_true_set)

    try:
        return len(tp) / (len(tp) + len(fp))
    except ZeroDivisionError:
        return 0.0


# --------------------------------------------------------------------------- #
def average_precision(y_true, y_pred):
    y_pred_set = set(y_pred)
    y_true_set = set(y_true)

    tp = y_pred_set.intersection(y_true_set)
    fn = y_true_set.difference(y_pred_set)

    p_sum = 0

    for match in tp:
        match_pos = np.argwhere(np.array(y_pred) == match)
        p_sum += precision(y_true, y_pred[:match_pos[0][0] + 1])

    _precision = precision(y_true, y_pred)
    for not_match in fn:
        p_sum += _precision

    try:
        return p_sum / len(y_true)
    except ZeroDivisionError:
        return 0.0


# --------------------------------------------------------------------------- #
def recall(y_true, y_pred):
    """
        Calculates recall score for a list of relevant documents and the groundtruth.

        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.

        Returns
        -------
        Score: float
            Recall = TP / (TP + FN)
    """
    y_true_set = set(y_true)
    y_pred_set = set(y_pred)

    tp = y_pred_set.intersection(y_true_set)
    fn = y_true_set.difference(y_pred_set)

    try:
        return len(tp) / (len(tp) + len(fn))
    except ZeroDivisionError:
        return 0.0


# --------------------------------------------------------------------------- #
def fscore(y_true, y_pred, beta=1.0):
    """
        Calculates f-measure for a list of relevant documents and the groundtruth.

        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.
        beta : float
            beta parameter weighting precision vs. recall

        Returns
        -------
        Score: float
            F-Measure = (1 + beta^2) \cdot \frac{Precision \cdot Recall}{beta^2 \cdot Precision+Recal}
    """

    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)

    try:
        return (1 + beta**2) * ((_precision * _recall) / ((beta**2 * _precision) + _recall))
    except ZeroDivisionError:
        return 0.0


# --------------------------------------------------------------------------- #
def precision_recall_fscore(y_true, y_pred, beta=1.0):
    return precision(y_true, y_pred), recall(y_true, y_pred), fscore(y_true, y_pred, beta)


# --------------------------------------------------------------------------- #
def plot_metric(metric_name, metric):
    plt.title(metric_name)
    plt.hist(metric)
    plt.show()


class RetrievalScorer:
    """
    Retrieval score system.
    Provides functions like RScore, Average Precision and Mean-Average-Precision.

    Attributes
    ----------
    retrieval_system : class object
           A Retrieval system. Must implement the abstract class InitRetrievalSystem.
    Methods
    -------
    rPrecision(y_true, query)
        Calculate the RScore.
    aveP(query, groundtruth)
        Calculate the average precision score for a query.
    MAP(queries, groundtruths)
        Calculate the mean average precision for a list of queries.

    """

    def __init__(self, system):
        """
        Initializes a RetrievalScorer class object

        Parameters
        ----------
        system : class object
            A retrieval system that implements InitRetrievalSystem.
        """
        self.retrieval_system = system

    # --------------------------------------------------------------------------- #
    def rPrecision(self, y_true, query):
        """
        Calculates the precision at R where R denotes the number of all relevant
        documents to a given query.

        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        query : str
            A query.

        Returns
        -------
        Score: float
            R-precision = TP / (TP + FP)
        """

        result = self.retrieval_system.retrieve_k(query, len(y_true))
        y_pred_set = set([res[0] for res in result])
        y_true_set = set(y_true)

        tp = y_pred_set.intersection(y_true_set)

        try:
            return len(tp) / len(y_true_set)
        except ZeroDivisionError:
            return 0.0

    # --------------------------------------------------------------------------- #
    def elevenPointAP(self, query, y_true):
        """
        Calculate the 11-point average precision score.

        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        query : str
            A query.

        Returns
        -------
        Tuple: (float, list, list)
            (11-point average precision score, recall levels, precision levels).
        """
        pass

    # --------------------------------------------------------------------------- #
    def MAP(self, queries, groundtruths):
        """
        Calculate the mean average precision.

        Parameters
        ----------
        groundtruths : list(list)
            A double nested list. Each entry contains a list of known relevant documents for a given query.
        queries : list(str)
            A list of queries. Each query maps exactly to one groundtruth list in groundtruths.

        Returns
        -------
        Score: float
            MAP = frac{1}{|Q|} \cdot \sum_{q \in Q} AP(q).
        """
        try:
            ap_sum = 0
            for i in range(len(queries)):
                ap_sum += average_precision(queries[i], groundtruths[i])
            return (1 / len(queries)) * ap_sum
        except ZeroDivisionError:
            return 0.0

    # --------------------------------------------------------------------------- #
    def pares_queries(self):
        query_map = {}
        with open(os.path.join(os.getcwd(), "data", configuration.QUERIES_FILE), 'r', encoding='utf8') as f:
            file = f.read()
            queries = file.split('\n')
        for query in queries:
            if query == '':
                continue
            query_parts = query.split('\t')
            qid, qtext = query_parts[0], query_parts[1]
            query_map[qid] = qtext

        return query_map

    # --------------------------------------------------------------------------- #
    def parse_qrel(self):
        relevance_map = {}
        with open(os.path.join(os.getcwd(), "data", configuration.RELEVANCE_FILE), 'r', encoding='utf8') as f:
            file = f.read()
            relevance_info = file.split('\n')

        for line in relevance_info:
            line_parts = line.split('\t')
            if line == '':
                continue
            qid, docid, relevance_lvl = line_parts[0], line_parts[2], line_parts[3]
            # TODO make use of relevance level
            docid = int(docid.replace('MED-', ''))
            try:
                relevance_map[qid].append(docid)
            except KeyError:
                relevance_map[qid] = [docid]

        return relevance_map

    # --------------------------------------------------------------------------- #
    def eval(self):
        q_map = self.pares_queries()
        rel_map = self.parse_qrel()

        list_precision = []
        list_recall = []
        list_f1 = []

        r_precision = []

        for qid, rel_docs in rel_map.items():
            result = self.retrieval_system.retrieve_k(q_map[qid], configuration.TOP_K_DOCS)
            r_precision.append(self.rPrecision(rel_docs, q_map[qid]))
            try:
                # tupel in case of retrieve_k()
                predicted = [res[0] for res in result]
            except Exception:
                predicted = result

            prec, rec, f1 = precision_recall_fscore(y_true=rel_docs, y_pred=predicted)
            print(f'{q_map[qid]} \n \t TRUE: {rel_docs} \n  \t PRED: {predicted} \n \t Precision: {prec} Recall: {rec} F1: {f1} \n')
            list_recall.append(rec)
            list_precision.append(prec)
            list_f1.append(f1)

        plot_metric("R-Precision", r_precision)
