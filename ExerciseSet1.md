
# Introduction to Data Science

# Exercise set 1

Student: Vladimir Dobrodeev

## Exercise 1

### Basics

Creating an array:


```python
import time
import sys

start = time.time()
a = [x for x in range(0, 10**7)]
b = [x for x in range(0, 10**7)]
elapsed = time.time() - start

print("Array size: %s MB. Array initialization has taken %.2f sec" % (sys.getsizeof(a) / (1024*1024), elapsed));
```

    Array size: 77 MB. Array initialization has taken 2.77 sec
    

Calculating a summ of arrays:


```python
def add_with_for(a, b):
    return [a[i] + b[i] for i in range(0, len(a))]

start = time.time()
c = add_with_for(a,b)
elapsed = time.time() - start

print("Calculating a summ took %.2f sec" % elapsed)
```

    Calculating a summ took 1.56 sec
    

Now let's initialize the array with NumPy:


```python
import numpy as np

start = time.time()
a = np.arange(0, 10 ** 7, dtype=np.int32)
elapsed = time.time() - start
print("Array size: %s MB. Array initialization has taken %.2f sec" % (sys.getsizeof(a) / (1024*1024), elapsed));

b = np.arange(0, 10 ** 7, dtype=np.int32)
```

    Array size: 38 MB. Array initialization has taken 0.09 sec
    


```python
start = time.time()
c = a + b
elapsed = time.time() - start

print("Calculating a summ took %.2f sec" % elapsed)
```

    Calculating a summ took 0.10 sec
    

So as a result NumPy array took two times less memory. The summ was calculated much faster. Speaking of memory it should be noticed, that used type is int32 (takes 4 bytes). And it also seems that at the current 64 bit machine, Python makes int as int64 (takes 8 bytes). THis is a source of difference. If we test the same code with int64 the result for NumPy will be different:


```python
start = time.time()
a = np.arange(0, 10 ** 7, dtype=np.int64)
elapsed = time.time() - start
print("Array size: %s MB. Array initialization has taken %.2f sec" % (sys.getsizeof(a) / (1024*1024), elapsed));

b = np.arange(0, 10 ** 7, dtype=np.int64)

start = time.time()
c = a + b
elapsed = time.time() - start

print("Calculating a summ took %.2f sec" % elapsed)
```

    Array size: 76 MB. Array initialization has taken 0.05 sec
    Calculating a summ took 0.05 sec
    

Unless size now is almost same as for normal list/array, the time of initialization and calculation is still better.

### Array manipulation

a)


```python
print np.arange(0, 100).reshape((10, 10))
```

    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]
     [30 31 32 33 34 35 36 37 38 39]
     [40 41 42 43 44 45 46 47 48 49]
     [50 51 52 53 54 55 56 57 58 59]
     [60 61 62 63 64 65 66 67 68 69]
     [70 71 72 73 74 75 76 77 78 79]
     [80 81 82 83 84 85 86 87 88 89]
     [90 91 92 93 94 95 96 97 98 99]]
    

b) 


```python
print np.fromfunction(lambda i, j: j % 2, shape=(10, 10), dtype=np.float)
```

    [[0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
     [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]]
    

c) 


```python
c = np.ones((10, 10))
np.fill_diagonal(c, 0)
print c
```

    [[0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 0. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 0. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 0. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 0. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 0. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 0. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 0. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 0. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]]
    

d) Note: we'll use array c from the previous example


```python
d = np.fliplr(c)
print d
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 0. 1.]
     [1. 1. 1. 1. 1. 1. 1. 0. 1. 1.]
     [1. 1. 1. 1. 1. 1. 0. 1. 1. 1.]
     [1. 1. 1. 1. 1. 0. 1. 1. 1. 1.]
     [1. 1. 1. 1. 0. 1. 1. 1. 1. 1.]
     [1. 1. 1. 0. 1. 1. 1. 1. 1. 1.]
     [1. 1. 0. 1. 1. 1. 1. 1. 1. 1.]
     [1. 0. 1. 1. 1. 1. 1. 1. 1. 1.]
     [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    

e)


```python
c = np.mat(c)
d = np.mat(d)
detC = np.linalg.det(c)
detD = np.linalg.det(d)
print("Determinant of C: %.2f" % detC)
print("Determinant of D: %.2f" % detD)
print("Multiplication: %.2f" % (detC * detD))
print("Matrix multiplication determinant: %.2f" % np.linalg.det(c * d))
```

    Determinant of C: -9.00
    Determinant of D: 9.00
    Multiplication: -81.00
    Matrix multiplication determinant: -81.00
    

So determinnat of multiplication is the same as multiplication of determinants. This holds not only for these matricies but for all matricies as it is a property of determinant.

### Slicing

a) Download a set and make sure, that its size correct:


```python
import sklearn.datasets

boston_dataset = sklearn.datasets.load_boston()
print boston_dataset.data.shape
```

    (506L, 13L)
    

So dataset size is correct

b) Rows with crime index > 1


```python
crim_index = np.where(boston_dataset.feature_names == "CRIM")[0][0]
rows = np.where(boston_dataset.data[:, crim_index] > 1)[0]
print("Total number of rows with such CRIM is %s which is %.2f%%" 
      % (rows.shape[0], float(boston_dataset.data.shape[0]) / rows.shape[0]))
print("Rows are:")
print(rows)
```

    Total number of rows with such CRIM is 173 which is 2.92%
    Rows are:
    [ 16  20  22  29  30  31  32  33  34 131 141 142 143 144 145 146 147 148
     149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166
     167 168 169 170 171 310 356 357 358 359 360 361 362 363 364 365 366 367
     368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385
     386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403
     404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421
     422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439
     440 441 442 443 444 446 447 448 449 450 451 452 453 454 455 456 457 458
     459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476
     477 478 479 480 481 482 483 484 485 486 487]
    

c) Rows with pupil-to-teach ratio from 16% to 18% (both excluded):


```python
ptratio_index = np.where(boston_dataset.feature_names == "PTRATIO")[0][0]
rows = np.where(np.logical_and(boston_dataset.data[:, ptratio_index] > 16, boston_dataset.data[:, ptratio_index] < 18))[0]
print "Total number of rows with such PTRATIO is %s which is %.2f%%" % (rows.shape[0], float(boston_dataset.data.shape[0]) / rows.shape[0])
print("Rows are:")
print(rows)
```

    Total number of rows with such PTRATIO is 100 which is 5.06%
    Rows are:
    [  1   2  41  42  43  44  45  46  47  48  49  50  51  52  53  55  56  65
      66  88  89  90  91 111 112 113 114 115 116 117 118 119 172 173 174 175
     176 177 178 179 180 181 182 183 184 185 186 199 200 216 217 218 219 220
     221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238
     239 240 241 242 243 254 255 274 275 276 277 278 287 288 289 301 302 303
     328 329 330 331 332 343 344 347 348 353]
    

d) Mean nitric oxides concentration for homes whose median price is more than $25000


```python
nox_index = np.where(boston_dataset.feature_names == "NOX")[0][0]
print "Mean nitric oxides concentration: %.3f" 
    % np.mean(boston_dataset.data[np.where(boston_dataset.target > 25)[0], nox_index])
```

    Mean nitric oxides concentration: 0.492
    

## Text data

For this exercise I used video games reviews. This is due to this dataset was faster to download. Code:


```python
import nltk
import re

# Define a stemmer
# This stemmer is described by ntlk web site as "better one"
stemmer = nltk.stem.snowball.SnowballStemmer("english")

# Prepare stemming function
# It should be vector to be used with NumPy arrays
stem_func = np.vectorize(lambda w: stemmer.stem(w))

# This function reads stop words
# Stop words are taken from the provided resource
def read_stop_words():
    stop_words_list = list()
    with open("stop-word-list.csv", "r") as stop_words_file:
        for csv_line in stop_words_file:
            for word in csv_line.split(","):
                stop_words_list.append(word.strip())
    # This fake stop word is required as theer might be multiple spaces between words
    # This will result into empty strings
    # Such strings could be easily removed while removing normal stop words
    stop_words_list.append("")
    stop_words_list = np.array(stop_words)
    return stop_words

stop_words = read_stop_words()

# Prepare a file for positive responses
positive_reviews = open("pos.txt", "w")
# Prepare a file for negative responses
negative_reviews = open("neg.txt", "w")
with open("Video_Games_5.json", "r") as json_file:
    for review_json in json_file:
        review = json.loads(review_json)
        # Reviews with grade 3 are avoided as texts of such reviews will not be needed anyway
        if review["overall"] <= 2 or review["overall"] >= 4:
            # Remove punctuation
            review_words = np.array(re.sub("[^a-z0-9 ]", "", review["reviewText"].lower()).split(" "))
            # Remove stop words
            # invert=True is required as otherwise everything EXCEPT stop words will be removed
            review_words = review_words[np.where(np.isin(review_words, stop_words, invert=True))]
            # Sometimes reviews contain only stop words
            # Most likely such review do not bring valuable information
            if review_words.shape[0] > 0:
                # Such reviews are avoided
                # For others the stemmer is applied
                review_words = stem_func(review_words)
                # And then, based on the grade, the review goes to a corresponding file
                if review["overall"] <= 2:
                    negative_reviews.write("%s\n" % " ".join(review_words))
                elif review["overall"] >= 4:
                    positive_reviews.write("%s\n" % " ".join(review_words))
negative_reviews.close()
positive_reviews.close()


```

There are some notes on this solution. First, it took a lot of time to complete (about 7 - 8 minutes). This is because fie was read sequentally. A good solution for this might be using Spark. Even locally it should allow to solve the problem faster. Spark is able to work with text files. Then, there are some reviews, where all words were stop words. An example of such review is "lol" (there is indeed such review). It is possble to guess, whether the author liked the game or not, by checking the grade. But the review itself does not contain any valuable information (it is not clear what was funny in the game).

It is not possible for now say, which words are actively used for positive or negative reviews. Quick glance highlights, that in positive reviews word "really" is used and for negative words like "install" and "research" seem to be quite widely used.
