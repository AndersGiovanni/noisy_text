"""
Classification using neural models.
Twitter Embeddings obtained from: https://nlp.stanford.edu/projects/glove/
"""

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER

import random
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from classify import encode_label, train_eval

# Fix seed for replicability
seed=103
random.seed(seed)
np.random.seed(seed)

def fileLoadPrep(file, embedding_file):
    """
    Load file and prepare data based on GloVe embeddings. 
    Tutorial followed: https://stackabuse.com/python-for-nlp-word-embeddings-for-deep-learning-in-keras/
    """
    # Load file
    df = pd.read_csv(file, sep="\t")
    df["Label"] = df["Label"].apply(lambda x: encode_label(x))
    
    y = df["Label"]

    # Define and fit word tokenizer
    word_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    word_tokenizer.fit_on_texts(df["Text"].values)

    # Vocab length
    vocab_length = len(word_tokenizer.word_index) + 1

    # Embed sentences
    embedded_sentences = word_tokenizer.texts_to_sequences(df["Text"].values)

    # Find max/mean/median tweet length
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(df["Text"].values, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    length_median = int(np.median([word_count(i) for i in df["Text"].values]))

    # Pad sentences
    X = tf.keras.preprocessing.sequence.pad_sequences(embedded_sentences, length_median, padding='post')

    # Prepare embeddings
    embeddings_dictionary = dict()
    glove_file = open(embedding_file, encoding="utf8")
    for line in tqdm(glove_file):
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_length, 100))
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    print("Done")

    return X, y, embedding_matrix, vocab_length, length_median


def makeModel(vocab_length, embedding_dim, embedding_matrix, input_dim):
    model = tf.keras.Sequential()
    embedding_layer = tf.keras.layers.Embedding(vocab_length, embedding_dim, weights=[embedding_matrix], input_length=input_dim, trainable=False)
    model.add(embedding_layer)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedFile", required=True, help = "Specify embedding file")
    #parser.add_argument()
    args = parser.parse_args()

    train_data_path = "data/train_lower_entities.tsv"
    test_data_path = "data/valid_lower_entities.tsv"
    embedding_file = args.embedFile

    X, y, embedding_matrix, vocab_length, input_dim = fileLoadPrep(train_data_path, embedding_file)

    model = makeModel(vocab_length, embedding_matrix.shape[1], embedding_matrix, input_dim)

    model.fit(X, y, epochs=100, verbose=1)

    loss, accuracy = model.evaluate(X, y, verbose=0)
    print('Accuracy: %f' % (accuracy*100))