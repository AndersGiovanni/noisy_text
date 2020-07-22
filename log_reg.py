__author__ = "AGMoller"

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import random
import os
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

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

def load_file(file, DictVect = False, tfidf = False, word_gram:str = "5", char_gram:str = "4"):
    """
    Load file and transform into correct format adapted from https://github.com/bplank/bleaching-text/
    
    If one wants to transform test data based on training data, a DictVectorizer
    based on training data must be given.
    """

    # Read file
    df = pd.read_csv(file, sep="\t")

    # Convert labels
    df["Label"] = df["Label"].apply(lambda x: encode_label(x))

    x = df["Text"].values
    y = df["Label"].values
        
    if DictVect == False:
        
        dictVectorizer = DictVectorizer()

        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram)
        x_dict = vectorizerWords.fit_transform(x)
        x_train = dictVectorizer.fit_transform(x_dict)

        print("Vocab size: ", len(dictVectorizer.vocabulary_))

        if tfidf == True:

            tfIdfTransformer = TfidfTransformer(sublinear_tf=True)

            x_train = tfIdfTransformer.fit_transform(x_train)

            return x_train, y, dictVectorizer, tfIdfTransformer

        else:

            return x_train, y, dictVectorizer, 0

    else:
        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram)
        x_dict = vectorizerWords.fit_transform(x)
        x_test = DictVect.transform(x_dict)

        print("Vocab size: ", len(DictVect.vocabulary_))

        if tfidf != False:

            x_test = tfidf.transform(x_test)

            return x_test, y, DictVect, tfidf

        else:

            return x_test, y, DictVect, 0

def train_eval(X_train, y_train, X_test, y_test):

    """
    Adapted from https://github.com/bplank/bleaching-text/
    Classifier has been changed from LinearSVC to Logistic Regression
    """

    classifier = LogisticRegression(n_jobs=-1)
    #classifier = LinearSVC()
    #classifier = MLPClassifier(hidden_layer_sizes=(100, 32, 1), random_state=seed)
    
    classifier.fit(X_train, y_train)
    print(classifier.classes_)

    y_predicted_test = classifier.predict(X_test)
    y_predicted_train = classifier.predict(X_train)

    accuracy_dev = accuracy_score(y_test, y_predicted_test)
    accuracy_train = accuracy_score(y_train, y_predicted_train)
    print("Classifier accuracy train: {0:.2f}".format(accuracy_train*100))


    print("===== dev set ====")
    print("Classifier: {0:.2f}".format(accuracy_dev*100))

    print(classification_report(y_test, y_predicted_test, digits=4))

    return f1_score(y_test, y_predicted_test, average="weighted"), accuracy_score(y_test, y_predicted_test), y_predicted_test


if __name__ == "__main__":

    print(os.listdir("data/"))

    wg = "5"
    cg = "4"

    X_train, y_train, dictvect, tfidf = load_file("data/train.tsv", word_gram=wg, char_gram=cg)
    X_dev, y_dev, _, _ = load_file("data/valid.tsv", dictvect, word_gram=wg, char_gram=cg)

    X = hstack((X_train, X_dev))
    #y = np.concatenate([y_train.todense(), y_dev.todense()])

    print(X_train.shape, y_train.shape)
    print(X_dev.shape, y_dev.shape)
    print(X.shape)

    #f1_test, acc_test, _ = train_eval(X_train, y_train, X_dev, y_dev)
    #print("weighted f1: {0:.1f}".format(f1_test * 100))
    #print("accuracy: {0:.1f}".format(acc_test * 100))
