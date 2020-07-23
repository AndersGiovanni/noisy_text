__author__ = "AGMoller"

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, regularizers

# Fix seed for replicability
seed=103
random.seed(seed)
np.random.seed(seed)

def encode_label(label):
    """
    Convert UNINFORMATIVE to 0 and INFORMATIVE to 1
    """
    if label == "UNINFORMATIVE": return 0
    else: return 1

def load2Files(file1, file2, DictVect = False, tfidf = False, word_gram:str = "5", char_gram:str = "4"):
    """
    This function combines two files. Used to make KFold including both training and val data.
    """


    # Read file
    df1 = pd.read_csv(file1, sep="\t")
    df2 = pd.read_csv(file2, sep="\t")

    # Convert labels
    df1["Label"] = df1["Label"].apply(lambda x: encode_label(x))
    df2["Label"] = df2["Label"].apply(lambda x: encode_label(x))

    x1 = df1["Text"].values
    x2 = df2["Text"].values
    y1 = df1["Label"].values
    y2 = df2["Label"].values

    # Combine files
    X = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))

    dictVectorizer = DictVectorizer()

    vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram)
    x_dict = vectorizerWords.fit_transform(X)
    x_train = dictVectorizer.fit_transform(x_dict)

    print("Vocab size: ", len(dictVectorizer.vocabulary_))

    if tfidf == True:

        tfIdfTransformer = TfidfTransformer(sublinear_tf=True)

        x_train = tfIdfTransformer.fit_transform(x_train)

        return x_train, y
    
    return x_train, y

def define_model(input_dim):
    """
    Define model architechture.
    Takes input dimension as input to specify shape of data.
    """

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation = "relu")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation = "relu")(x)
    x = layers.Dropout(0.2)(x)
    o = layers.Dense(1, activation = "sigmoid")(x)

    model = tf.keras.Model(inputs = inputs, outputs = o)
    opt = tf.keras.optimizers.Adagrad(0.001)
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["binary_accuracy"])

    return model

def kfold(X, y, input_dim):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    f1 = []
    split = 1

    for train, test in skf.split(X,y):

        print("Split: {}".format(split))
        split += 1 

        # Extract train and test
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        print("X_train shape: {}".format(X_train.shape))

        model = define_model(input_dim)

        model.fit(X_train, y_train,
                            epochs = 10,
                            batch_size = 64)

        y_predicted_test = model.predict(X_test)
        y_predicted_test = [1 if i > 0.5 else 0 for i in y_predicted_test]

        f1.append(f1_score(y_test, y_predicted_test, average="weighted"))

    print("Avg F1: {}\n Individual runs: {}".format(np.mean(f1), f1))

if __name__ == "__main__":

    X, y = load2Files("data/train.tsv", "data/valid.tsv")

    input_dim = X.shape[1] 

    kfold(X, y, input_dim)