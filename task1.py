import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
import time

filename = "passage-collection.txt"


# preprocessing file
def parsing_file(filepath):
    with open(filepath, "r", encoding='utf-8') as file:
        data = file.readlines()

    # delete white spaces and all work to lower case
    return [line.strip().lower() for line in data]

def preprocessing(data, remove_stop_words=False):
    # parse file and tokenize

    processed = []
    punctuation = string.punctuation
    stopword = stopwords.words("english")
    snowball = SnowballStemmer('english')
    for line in data:
        line = word_tokenize(line)
        new_line = []
        for word in line:
            if word in punctuation:
                continue
            word = snowball.stem(word)
            if word.isdigit():
                continue
            if remove_stop_words and word in stopword:
                continue

            new_line.append(word)
        processed.append(new_line)

    return processed


def count_feq(tokens, sort=False):
    dictionary = {}
    for line in tokens:
        for token in line:
            if token in dictionary:
                dictionary[token] = dictionary[token] + 1
            else:
                dictionary[token] = 1
    if sort:
        dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return dictionary


def zipfian(k, N, s):
    dividend = np.arange(1.0, N + 1) ** -s
    divisor = sum([x ** -s for x in np.arange(1.0, N + 2)])
    return dividend / divisor


if __name__ == "__main__":
    data = parsing_file(filename)

    tokens = preprocessing(data)
    tokens_remove_stop = preprocessing(data, remove_stop_words=True)

    # processed_tokens = preprocessing(filename)
    word_occurance = count_feq(tokens, sort=True)

    words = len(word_occurance)
    total_terms = sum([x[1] for x in word_occurance])

    print(total_terms)

    probability = [x[1] / total_terms for x in word_occurance]

    rankings = np.arange(1, words + 1)
    zipf = zipfian(words, words, 1)

    plt.plot(rankings, zipf, label="zipf")
    plt.plot(rankings, probability,label="processed data")
    plt.xlabel('frequency ranking')
    plt.ylabel('probability of occurrence')
    plt.legend()
    plt.savefig('task1_figure1.png')
    plt.show()



    word_occurance_remove_stop = count_feq(tokens_remove_stop, sort=True)

    words_remove = len(word_occurance_remove_stop)

    total_terms_remove = sum([x[1] for x in word_occurance])
    rankings_remove = np.arange(1, words_remove + 1)

    zip_remove = zipfian(words_remove, words_remove, 1)

    probability_remove_stop = [x[1] / total_terms_remove for x in word_occurance_remove_stop]
    print("average passage lenth:" ,sum([x[1] for x in word_occurance_remove_stop]) / words_remove)

    print("original vocabulary size:", len(word_occurance))
    print("vocabulary size without stop words:", len(word_occurance_remove_stop))

    plt.loglog(rankings, zipf,label="zipf")

    plt.loglog(rankings, probability, label="contains stop word")
    plt.loglog(rankings_remove, probability_remove_stop, label="stop word removed")

    plt.xlabel('frequency ranking(log scale)')
    plt.ylabel('probability of occurrence(log scale)')
    plt.legend()
    plt.savefig('task1_figure2.png')
    plt.show()


