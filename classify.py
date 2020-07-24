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
from sklearn.model_selection import StratifiedKFold


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

def load_file(file, DictVect = False, tfidf = False, tfIdfTransformer = None, word_gram:str = "5", char_gram:str = "4"):
    """
    Load file and transform into correct format adapted from https://github.com/bplank/bleaching-text/
    
    If one wants to transform test data based on training data, a DictVectorizer
    based on training data must be given.

    # fixed tfidf transformer
    """

    # Read file
    df = pd.read_csv(file, sep="\t")

    # Convert labels
    df["Label"] = df["Label"].apply(lambda x: encode_label(x))

    x = df["Entities"].values
    x = df["Entities_Details"].values

    #x = df["Text"].values
    y = df["Label"].values

    if DictVect == False:

        dictVectorizer = DictVectorizer()

        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram, binary=False)
        x_dict = vectorizerWords.fit_transform(x)
        print("first instance as features:", x_dict[0])
        x_train = dictVectorizer.fit_transform(x_dict)
        print("Vocab size train: ", len(dictVectorizer.vocabulary_))

        if tfidf == True:

            tfIdfTransformer = TfidfTransformer(sublinear_tf=True)

            x_train = tfIdfTransformer.fit_transform(x_train)

            return x_train, y, dictVectorizer, tfIdfTransformer

        else:

            return x_train, y, dictVectorizer, tfIdfTransformer

    else:
        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram)
        x_dict = vectorizerWords.fit_transform(x)
        x_test = DictVect.transform(x_dict)

        print("Vocab size: ", len(DictVect.vocabulary_))

        if tfidf != False:

            x_test = tfIdfTransformer.transform(x_test)

            return x_test, y, DictVect, tfidf

        else:

            return x_test, y, DictVect, tfIdfTransformer



def train_eval(classifier, X_train, y_train, X_test, y_test):

    """
    Adapted from https://github.com/bplank/bleaching-text/
    Classifier has been changed from LinearSVC to Logistic Regression
    """

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

    train_data_path = "data/train_lower_entities.tsv"
    test_data_path = "data/valid_lower_entities.tsv"
    wg = "1-3"
    cg = "4"
    wg = "1"
    cg = "0"
    use_tfidf = True

    #classifier = LogisticRegression(n_jobs=-1, max_iter= 10000)
    classifier = LinearSVC(penalty='l2', loss='squared_hinge', \
                           dual=True, tol=0.0001, C=1.0, multi_class='ovr', \
                           fit_intercept=True, intercept_scaling=1, class_weight=None)
    # classifier = MLPClassifier(hidden_layer_sizes=(100, 32, 1), random_state=seed)
    print(classifier)
    X_train, y_train, dictvect, tfidfvect = load_file(train_data_path, tfidf=use_tfidf, tfIdfTransformer=None, word_gram=wg, char_gram=cg)
    print(tfidfvect)
    X_dev, y_dev, _, _ = load_file(test_data_path, DictVect=dictvect, tfidf=use_tfidf, tfIdfTransformer=tfidfvect, word_gram=wg, char_gram=cg)

    # X, y = load2Files("data/train.tsv", "data/valid.tsv")

#    kfold(X, y)

    f1_test, acc_test, _ = train_eval(classifier, X_train, y_train, X_dev, y_dev)
    print("weighted f1: {0:.1f}".format(f1_test * 100))
    print("accuracy: {0:.1f}".format(acc_test * 100))
