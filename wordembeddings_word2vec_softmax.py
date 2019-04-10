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


    batch = np.ndarray(shape=batch_size, dtype=np.int32 )
    # batch holds the input words and its size should be equal to batch size
    # batch = [1, 2, 3, .... batch_size]
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # the label contains the indexes of the target predicted words
    # labels = [[1], [2], [3], ..., [batch_size]]
    span = 2 * skip_window + 1  # [skip window input_word skip_window]
    # span is the total size of context window

    # A deque is double-ended queue which supports memory efficient appends
    # and pops from each side
    buffer = collections.deque(maxlen=span)
    # text fragment within the context window is stored in a deque with the maximum size of the context window

    # Initialize the deque with the first words in the deque
    for _ in range(span):
        buffer.append(word_indexes[global_index])
        global_index = (global_index + 1) % len(word_indexes)  # reset the global index to the beginning

    for i in range(batch_size // num_skips):
        # each input word will be used to predict num_skips target words
        target = skip_window
        target_to_avoid = [skip_window]  # do not chose the input word

        for j in range(num_skips):  # selecting words at random from the context window
            target = random.randint(0, span-1)
            target_to_avoid.append(target)  # choosing different context word each time
            batch[i * num_skips + j] = buffer[skip_window]  # input word
            label[i * num_skips + j, 0] = buffer[target]  # context words

        # appending a word to the deque and removes a word from the beginning
        buffer.append(word_indexes[global_index])
        global_index = (global_index+1) % len(word_indexes)

    # ensure the words at the end of the batch are included in the next batch
    global_index = (global_index + len(word_indexes) - span) % len(word_indexes)

    return batch, label


batch, labels = generate_batch(word_indexes, 10, 2, 5)
print(batch, labels)
for i in range(9):  # print the word and corresponding target words rather than word indexes
    print(reversed_dictionary[batch[i]], ":", reversed_dictionary[labels[i][0]])


 #**** Initialize some variables to build and train the skip-gram model (Construct the Neural Network) ****

# Reset the global index
global_index = 0

# validating the embeddings of words that are similar are close together
valid_size = 16
valid_window = 100

# choosing at random 16 of the top 100 most frequently occurring words and find the closest neighbour
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

batch_size = 128
embedded_size = 50  # the number of dimension that the word embedding will have (number of neuron=50)
skip_window = 2
num_skips = 2

tf.reset_default_graph()
# at every iteration 128 data is trained with corresponding labels
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# validation set
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# *** Embeddings are word representations that will be generated by word2vec

# Initialize a variable to hold the embeddings and embeddings for
# specific words can be accessed using *tf.nn.embedding_lookup*

# embedding is generated by training dataset
embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embedded_size], -1.0, 1.0))  # 50 dimension for every word

# train_inputs contains unique word indexes in the batch and lookup in the embedding matrix
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the softmax (A NN layer with no activation function)
weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embedded_size], stddev=1.0/math.sqrt(embedded_size)))
bias = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
hidden_out = tf.matmul(embed, tf.transpose(weights)) + bias

# converting the labels of training to one_hot to use it with softmax prediction layer
train_one_hot = tf.one_hot(train_labels, VOCABULARY_SIZE)
# Calculate the cross-entropy (loss function for Softmax prediction)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,labels=train_one_hot))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# ### Checking the wordtovec algorithm works correctly
# ### Normalize the embeddings vector to calculate cosine similarity between words
# *normalized_vector = vector / L2 norm of vector*

l2_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings/l2_norm  # find the cosin similarity of the words in validation dataset

# #### Look up the normalized embeddings of the words we use to validate our model
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)  # shape = (16,50)

# #### Find the cosine similarity of the validation words against all words in our vocabulary
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()
num_steps = 20001 # number of epoch

with tf.Session() as session:
    init.run()
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            word_indexes, batch_size,num_skips,skip_window)
        feed_dic = {train_inputs:batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer,loss], feed_dict=feed_dic)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000

            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average Loss at Step", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reversed_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reversed_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print("\n")
        final_embeddings = normalized_embeddings.eval()
