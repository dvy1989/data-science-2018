import json
import re
import sys
import time

import nltk
import sklearn.datasets

import numpy as np


def add_with_for(a, b):
    return [a[i] + b[i] for i in range(0, len(a))]


def basics():
    a = [x for x in range(0, 10 ** 7)]
    print "Simple list/array: %s" % (sys.getsizeof(a) / (1024 * 1024))
    b = [x for x in range(0, 10 ** 7)]
    start = time.time()
    c = add_with_for(a, b)
    print "Elapsed %s seconds" % (time.time() - start)
    print "Simple list/array: %s" % (sys.getsizeof(c) / (1024 * 1024))

    a = np.arange(0, 10 ** 7, dtype=np.int64)
    print "Numpy arange %s:" % (sys.getsizeof(a) / (1024 * 1024))
    b = np.arange(0, 10 ** 7, dtype=np.int64)
    start = time.time()
    c = a + b
    print "Elapsed %s seconds" % (time.time() - start)
    print c
    print "Numpy arange: %s" % (sys.getsizeof(c) / (1024 * 1024))

    a = np.arange(0, 10 ** 7, dtype=np.int32)
    print "Numpy arange %s:" % (sys.getsizeof(a) / (1024 * 1024))
    b = np.arange(0, 10 ** 7, dtype=np.int32)
    start = time.time()
    c = a + b
    print "Elapsed %s seconds" % (time.time() - start)
    print c
    print "Numpy arange: %s" % (sys.getsizeof(c) / (1024 * 1024))

    a = np.linspace(0, 10 ** 7 - 1, num=10 ** 7, dtype=np.int64)
    print "Numpy linspace: %s" % (sys.getsizeof(a) / (1024 * 1024))
    b = np.linspace(0, 10 ** 7 - 1, num=10 ** 7, dtype=np.int64)
    start = time.time()
    c = a + b
    print "Elapsed %s seconds" % (time.time() - start)
    print c
    print "Numpy linspace: %s" % (sys.getsizeof(c) / (1024 * 1024))

    a = np.linspace(0, 10 ** 7 - 1, num=10 ** 7, dtype=np.int32)
    print "Numpy linspace: %s" % (sys.getsizeof(a) / (1024 * 1024))
    b = np.linspace(0, 10 ** 7 - 1, num=10 ** 7, dtype=np.int32)
    start = time.time()
    c = a + b
    print "Elapsed %s seconds" % (time.time() - start)
    print c
    print "Numpy linspace: %s" % (sys.getsizeof(c) / (1024 * 1024))


def stepA():
    print np.arange(0, 100).reshape((10, 10))


def stepB():
    print np.fromfunction(lambda i, j: j % 2, shape=(10, 10), dtype=np.float)


def stepC():
    x = np.ones((10, 10))
    np.fill_diagonal(x, 0)
    return x


def stepD():
    return np.fliplr(stepC())


def arrayManipulation():
    stepA()
    stepB()
    c = stepC()
    d = stepD()
    print c
    print d
    c = np.mat(c)
    d = np.mat(d)
    detC = np.linalg.det(c)
    detD = np.linalg.det(d)
    print "Determinant of C: %s" % detC
    print "Determinant of D: %s" % detD
    print "Multiplication: %s" % (detC * detD)
    print c * d
    print "Matrix multiplication determinant: %s" % np.linalg.det(c * d)


def slicing():
    boston_dataset = sklearn.datasets.load_boston()
    print boston_dataset.data.shape
    print boston_dataset.feature_names
    crim_index = np.where(boston_dataset.feature_names == "CRIM")[0][0]
    rows = np.where(boston_dataset.data[:, crim_index] > 1)[0]
    print "Total number of rows with such CRIM is %s which is %.2f%%" % (rows.shape[0], float(boston_dataset.data.shape[0]) / rows.shape[0])
    ptratio_index = np.where(boston_dataset.feature_names == "PTRATIO")[0][0]
    rows = np.where(np.logical_and(boston_dataset.data[:, ptratio_index] > 16, boston_dataset.data[:, ptratio_index] < 18))[0]
    print "Total number of rows with such PTRATIO is %s which is %.2f%%" % (rows.shape[0], float(boston_dataset.data.shape[0]) / rows.shape[0])
    nox_index = np.where(boston_dataset.feature_names == "NOX")[0][0]
    print "%.3f" % np.mean(boston_dataset.data[np.where(boston_dataset.target > 25)[0], nox_index])


def exercise2():
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stem_func = np.vectorize(lambda w: stemmer.stem(w))
    stop_words = list()
    with open("stop-word-list.csv", "r") as stop_words_file:
        for csv_line in stop_words_file:
            for word in csv_line.split(","):
                stop_words.append(word.strip())
    stop_words.append("")
    stop_words = np.array(stop_words)
    positive_reviews = open("pos.txt", "w")
    negative_reviews = open("neg.txt", "w")
    start = time.time()
    with open("Automotive_5.json", "r") as json_file:
        for review_json in json_file:
            review = json.loads(review_json)
            if review["overall"] <= 2 or review["overall"] >= 4:
                review_words = np.array(re.sub("[^a-z0-9 ]", "", review["reviewText"].replace("\n", "").lower()).split(" "))
                review_words_without_stop_words = review_words[np.where(np.isin(review_words, stop_words, invert=True))]
                if review_words_without_stop_words.shape[0] > 0:
                    stemmed_words = stem_func(review_words_without_stop_words)
                    if review["overall"] <= 2:
                        negative_reviews.write("%s\n" % " ".join(stemmed_words))
                    elif review["overall"] >= 4:
                        positive_reviews.write("%s\n" % " ".join(stemmed_words))
    negative_reviews.close()
    positive_reviews.close()
    elapsed = time.time() - start
    print(elapsed)


if __name__ == "__main__":
    # exercise1()
    # exercise2()
    # slicing()
    exercise2()
