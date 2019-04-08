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

# ****************************** Implement Word Embeddings using skip-grams ******************************


# Download Zip file which contains the IMDB reviews and name it as SampleText.zip and it is in our local machine
def maybe_download(url_path, expected_bytes):
    # check that it has been downloaded or not! if not use urlertrieve func from urllib
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


# Building a dataset in a format that is useful to generate Word2Vec embedding
def build_dataset(words, n_words): # n_words: Use top N frequently used word
    word_counts = [['UNKNOWN', -1]]  # word_counts contains the words and the frequency of the word
                                    # the word which is not in the top n frequency will be added to unknown count
    counter = collections.Counter(words)
    word_counts.extend(counter.most_common(n_words-1)) # Top n frequently used word
    dictionary = dict()
    # Assign unique indexes to words in the dic, the most common words have the lowest index
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
for key in random.sample(list(reversed_dictionary), 10):
    print(key, ":", reversed_dictionary[key])

del vocabulary
global_index = 0


# generate_batch function return a new batch of data for every iteration
def generate_batch(word_indexes, batch_size, num_skips, skip_window):
    # word_indexes is the text of the entire document during training using a word's unique indexes
    # the num_skips represents the number of words that we chose from the context window of any input word
    # num_skip is the number of words to pick at random from the skip window
    # skip_window is the number of neighbors that is considered to the left and to the right of the skip gram model

    global global_index
    # the global index will keep track of where we are in the document as the context window slides over the text

    assert batch_size % num_skips == 0
    # in any batch every input word will appear num_skips number of times once with every target word

    assert num_skips <= 2 * skip_window
    # make sure the context window is big enough to pick num_skip words

    batch = np.ndarray(shape=(batch_size), dtype=np.int32 )
    # batch holds the iput words and its size should be equal to batch size

    label = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    # the label contains the indexes of the target predicted words

    span = 2 * skip_window + 1  # [skip window input_word skip_window]
    # span is the total size of context window

    buffer = collections.deque(maxlen=span)
    # text fragment within the context window is stored in a deque with the maximum size of the context window

    for _ in range(span):
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes) # reset the global index to the beginning

    for i in range(batch_size // num_skips):
        # each input word will be used to predict num_skips target words
        target = skip_window
        target_to_avoid = [skip_window]  # do not chose the input word

        for j in range(num_skips): # selecting words at random from the context window
            target = random.randint(0, span-1)
            target_to_avoid.append(target)  # choosing different context word each time
            batch[i * num_skips + j] = buffer[skip_window]  # input word
            label[i * num_skips + j, 0] = buffer[target] # context words

        # appending a word to the deque and removes a word from the beginning
        buffer.append(word_indexes[global_index])
        global_index = (global_index+1) % len(word_indexes)

    # ensure the words at the end of the batch are included in the next batch
    global_index = (global_index + len(word_indexes) - span) % len(word_indexes)

    return batch, label


batch , labels = generate_batch(word_indexes, 10, 2, 5)
print(batch, labels)
for i in range(9):  # print the word and corresponding target words rather than word indexes
    print(reversed_dictionary[batch[i]], ":", reversed_dictionary[labels[i][0]])

