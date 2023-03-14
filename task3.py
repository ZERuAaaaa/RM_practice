import math

import numpy as np
from nltk import FreqDist

import task1
import json
import pandas as pd


# calculate tfidf of given inverted_index
def tfidf(inverted_index, N):
    tfidf_dict = {}
    tf_dict = {}

    for token, payloads in inverted_index.items():
        idf = math.log10(N / len(payloads))
        for pid, tf in payloads.items():
            tfidf = inverted_index[token][pid] * idf

            if pid in tfidf_dict.keys():
                tfidf_dict[pid][token] = tfidf
                tf_dict[pid][token] = tf
            else:
                tfidf_dict[pid] = {token: tfidf}
                tf_dict[pid] = {token: tf}

    return tfidf_dict, tf_dict


# invert_index for query
def invert_index(table):
    inver_dict = {}
    for line in table.iterrows():
        current_line = line[1]
        for token in current_line['query'].keys():
            if token in inver_dict.keys():
                inver_dict[token].update({current_line['qid']: current_line['query'][token]})
            else:
                inver_dict[token] = {current_line['qid']: current_line['query'][token]}
    return inver_dict

# preprocess for query
def process_query(queries_data):
    # preprocess query
    query_length = len(queries_data)
    processed_query = task1.preprocessing(queries_data['query'], remove_stop_words=True)
    queries_data['query'] = processed_query

    queries_data['query'] = [FreqDist(line) for line in queries_data['query']]
    # calculate query inverted_index
    inverted_index_query = invert_index(queries_data)
    return query_length, inverted_index_query


def cos_similarity(tfidf_passage_dict, tfidf_query_dict, data, queries_data):
    cos_similarity_dict = {}

    for qid in queries_data['qid']:
        cos_similarity_dict[qid] = {}

        pid_q = data[data['qid'] == qid]
        tfidf_query = tfidf_query_dict[qid]
        for pid in pid_q['pid']:

            tfidf_passage = tfidf_passage_dict[str(pid)]

            inter = list(set(tfidf_passage.keys()).intersection(set(tfidf_query.keys())))

            inner_product = 0.0

            for key in inter:
                inner_product += tfidf_passage[key] * tfidf_query[key]

            norm_q = np.linalg.norm(list(tfidf_query.values()))
            norm_p = np.linalg.norm(list(tfidf_passage.values()))
            cos_similarity_dict[qid][pid] = inner_product / (norm_p * norm_q)

    return cos_similarity_dict


def flatten_dict(dictionary):
    keys = []
    for key, item in dictionary.items():
        keys.append(key)
        dictionary[key] = sorted(list(item.items()), key=lambda item: item[1], reverse=True)[:100]
    flatten = []
    for key in keys:
        for item in dictionary[key]:
            flatten.append([key, item[0], item[1]])
    flatten = pd.DataFrame(flatten, columns=['qid', 'pid', 'score'])
    return flatten


def BM25(data, queries_data, passage_tf, query_tf, inverted_index):
    k1 = 1.2
    k2 = 100
    b = 0.75
    dl, avdl = {}, 0
    bm25 = {}
    N = len(passage_tf)

    def cal_bm25(qf, K, n, pf):
        temp = np.log((N - n + 0.5) / (n + 0.5))
        temp *= (((k1 + 1) * pf) / (K + pf))
        temp *= (((k2 + 1) * qf) / (k2 + qf))
        return temp

    # calculate dl and avdl
    for pid, payload in passage_tf.items():
        dl[pid] = sum(payload.values())
        avdl += sum(payload.values())
    avdl /= len(dl)
    for qid in queries_data['qid']:
        bm25[qid] = {}
        pid_q = data[data['qid'] == qid]
        for pid in pid_q['pid']:
            query = query_tf[qid]
            passage = passage_tf[str(pid)]
            inter = list(set(query.keys()).intersection(set(passage.keys())))
            score_q_p = 0
            for token in inter:
                qf = query_tf[qid][token]
                pf = passage[token]
                K = k1 * ((1 - b) + ((b * dl[str(pid)]) / avdl))
                n = len(inverted_index[token])
                score_q_p += cal_bm25(qf, K, n, pf)
            bm25[qid][pid] = score_q_p

    return bm25

def parse_csv(to_flat,filename):
    flatten = flatten_dict(to_flat)
    flatten = pd.DataFrame(flatten)
    flatten.to_csv(filename, index=False, header=False)

if __name__ == "__main__":
    # load data
    data_filename = "candidate-passages-top1000.tsv"
    inverted_index_filename = "inverted_index.txt"
    query_filename = "test-queries.tsv"

    data = pd.read_csv(data_filename, sep="\t", names=['qid', 'pid', 'query', 'passage'])
    queries_data = pd.read_csv(query_filename, sep="\t", names=['qid', 'query'])
    inverted_index = json.loads(open(inverted_index_filename).read())

    query_length, inverted_index_query = process_query(queries_data)

    N = len(data)

    # calculate query and passage tfidf
    tfidf_passage_dict, tf_passage_dict = tfidf(inverted_index, N)
    tfidf_query_dict, tf_query_dict = tfidf(inverted_index_query, query_length)

    similarity = cos_similarity(tfidf_passage_dict, tf_query_dict, data, queries_data)
    parse_csv(similarity,"tfidf.csv")

    bm25 = BM25(data, queries_data, tf_passage_dict, tf_query_dict, inverted_index)
    parse_csv(bm25,"bm25.csv")
