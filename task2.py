import json

import numpy as np
import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import time

import task1


def invert_index(table):
    inver_dict = {}
    for line in table.iterrows():
        current_line = line[1]
        for token in current_line['passage'].keys():
            if token in inver_dict.keys():
                inver_dict[token].update({current_line['pid']: current_line['passage'][token]})
            else:
                inver_dict[token] = {current_line['pid']: current_line['passage'][token]}
    return inver_dict


if __name__ == "__main__":
    filename = "candidate-passages-top1000.tsv"

    data = pd.read_table(filename, sep='\t', names=['qid', 'pid', 'query', 'passage'])
    data.drop_duplicates(subset=['query', 'passage'], keep='last', inplace=True)
    data = data.reset_index(drop=True)

    result = task1.preprocessing(data['passage'], remove_stop_words=True)
    data['passage'] = result

    data['passage'] = [FreqDist(line) for line in data['passage']]

    input_table = data[['pid', 'passage']]

    inverted_index = invert_index(input_table)

    with open("inverted_index.txt", 'w') as writeFile:
        writeFile.write(json.dumps(inverted_index))
