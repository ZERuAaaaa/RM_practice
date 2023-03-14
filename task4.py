import json

import numpy as np
import pandas as pd

import task3


def language_model(data, queries_data, inverted_index, passage_tf, query_tf):
    e = 0.1
    mu = 50
    V = len(inverted_index)
    vocabulary_total = [[x for x in y.values()] for y in inverted_index.values()]
    V_t = sum([len(x) for x in vocabulary_total])

    Laplace = {}
    Lidstone = {}
    Dirichlet = {}
    def Lap(m,D,V):
        return np.log((m + 1) / (D + V))

    def Lid(m,D,V):
        return np.log((m + e) / (D + e * V))

    def Dir(D,m,w_c):

        term1 = D / (D+mu) * (m / D)
        term2 = mu / (D + mu) * w_c if w_c != 0 or D == 0 else 0
        return np.log(term1 + term2)

    for qid in queries_data['qid']:
        Laplace[qid] = {}
        Lidstone[qid] = {}
        Dirichlet[qid] = {}
        for pid_q in data[data['qid'] == qid]['pid']:
            D = sum(passage_tf[str(pid_q)].values())

            for token in query_tf[qid].keys():
                m = passage_tf[str(pid_q)].get(token, 0)
                w_c = len(inverted_index[token].values()) / V_t if token in inverted_index.keys() else 0

                Laplace[qid][pid_q] = Laplace[qid].get(pid_q, 0) + Lap(m,D,V)
                Lidstone[qid][pid_q] = Lidstone[qid].get(pid_q, 0) + Lid(m,D,V)
                Dirichlet[qid][pid_q] = Dirichlet[qid].get(pid_q,0) + Dir(D,m,w_c)


    return Laplace, Lidstone, Dirichlet



if __name__ == "__main__":
    data_filename = "candidate-passages-top1000.tsv"
    query_filename = "test-queries.tsv"
    inverted_index_filename = "inverted_index.txt"

    data = pd.read_csv(data_filename, sep="\t", names=['qid', 'pid', 'query', 'passage'])
    queries_data = pd.read_csv(query_filename, sep="\t", names=['qid', 'query'])

    inverted_index = json.loads(open(inverted_index_filename).read())

    query_length, inverted_index_query = task3.process_query(queries_data)

    N = len(data)

    # calculate query and passage tfidf
    tfidf_passage_dict, tf_passage_dict = task3.tfidf(inverted_index, N)
    tfidf_query_dict, tf_query_dict = task3.tfidf(inverted_index_query, query_length)

    Laplace, Lidstone, Dirichlet = language_model(data, queries_data, inverted_index, tf_passage_dict, tf_query_dict)

    task3.parse_csv(Laplace,"laplace.csv")
    task3.parse_csv(Lidstone, "lidstone.csv")
    task3.parse_csv(Dirichlet, 'dirichlet.csv')

