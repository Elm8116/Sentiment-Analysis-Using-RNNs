import collections
import math
import os
import random
import zipfile
from six.moves import urllib
from six.moves import xrange
import numpy as np
import tensorflow as tf
DOWNLOAD_FILENAME = 'SampleText.zip'


# Download Zip file which contains the IMDB reviews and name it as SampleText.zip and it is in our local machine
def maybe_download(url_path, expected_bytes):
    # check that it has been downloaded or not! if not use urlertrieve func from urllib
    # check if a path exists
    if not os.path.exists(DOWNLOAD_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path,DOWNLOAD_FILENAME)
    statinfo = os.stat(DOWNLOAD_FILENAME)
    if statinfo.st_size == expected_bytes:
        print("found: ", url_path)
        print("download file: ", DOWNLOAD_FILENAME)
    else:
        print(statinfo.st_size)


# Extract the contents of the SampleText.zip file and parse it into individual words
def read_words():
    with zipfile.ZipFile(DOWNLOAD_FILENAME) as f:
        # get the first file with its content
        firstfile = f.namelist()[0]
        # read the first file and convert it to the string format
        filestring = tf.compat.as_str(f.read(firstfile))
        # Extarct the individual words from this file string
        words = filestring.split()
    return words


URL_PATH = 'http://mattmahoney.net/dc/text8.zip'
FILESIZE = 31344016
maybe_download(URL_PATH,FILESIZE)

# Vocabulary contains all the words from the input dataset
vocabulary = read_words()
print(len(vocabulary))
print(vocabulary[:25])

# Building a dataset in a format that is useful to generate Word2Vec embedding
# n_words: Use top N frequently used word

def build_dataset(words, n_words):
    word_counts = [['UNKNOWN', -1]]  # the words and the frequency of the word
                                    # and word which is not in the top n frequency will be added to unknown count
    counter = collections.Counter(words)
    word_counts.extend(counter.most_common(n_words-1))
    print(word_counts)
    dictionary = dict()

    # higher frequency words will have lower index value and we store a mapping of a word to its unique index in the dic
    # Assign unique indexes to words, the most common words have the lowest index
    for word, _ in word_counts:
        dictionary[word] = len(dictionary)  # the len of dic is increasing by one at each step
    word_indexes = list()
    unknown_count = 0
    for word in words:  # the occurrence of other words outside of the top n will be added to the unknown  count
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unknown_count += 1
        word_indexes.append(index)

    word_counts[0][1] = unknown_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return word_counts, word_indexes, dictionary, reversed_dictionary


# building a dataset
VOCABULARY_SIZE = 5000
word_counts, word_indexes, dictionary, reversed_dictionary = build_dataset(vocabulary, VOCABULARY_SIZE)

print(word_counts[:10])
print(word_indexes[:10])

import random

for key in random.sample(list(dictionary), 10):
    print(key, ":", dictionary[key])

for key in random.sample(list(reversed_dictionary), 10):
    print(key, ":", reversed_dictionary[key])

del vocabulary


# Global index into words maintained across batches
# the global index will keep track of where we are in the doc as the contex window slides over the text
global_index = 0

# return  a new batch of data for every iteration
# word_indexes is the text of the entire document during training using a word's unique indexes
# the num_skips represents the number of words that we chose from the context window of any input word
# num_skip is the number of words to pick at random from the skip window
# skip_window is the number of neighbors that we want to consider to the left and to the right of the  skip gram model
# for example if the skip_window=3, it means that look at three words on either side of the input word
def generate_batch(word_indexes, batch_size, num_skips, skip_window):
