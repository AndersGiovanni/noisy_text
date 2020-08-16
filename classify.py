__author__ = "AGMoller"

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER

import random
import os
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
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

def load_file(file, feat, DictVect = False, tfidf = False, tfIdfTransformer = None, word_gram:str = "0", char_gram:str = "0", wordpiece_gram:str = "0"):
    """
    Load file and transform into correct format adapted from https://github.com/bplank/bleaching-text/
    
    If one wants to transform test data based on training data, a DictVectorizer
    based on training data must be given.

    # fixed tfidf transformer
    """

    # Read file
    df = pd.read_csv(file, sep="\t")

    #print(df.info())

    # Convert labels
    df["Label"] = df["Label"].apply(lambda x: encode_label(x))
    df1 = df[["Text", "Entities_Details", "Entities","Label"]]
    df1["Combined"] = df1[df1.columns[:-1]].apply(lambda x: ' __NEW_FEATURE__ '.join(x.dropna().astype(str)),axis=1)
    #x = df["Entities"].values
    #x = df["Entities_Details"].values
    #x = df[["Text","Entities"]].values

    #print("Hej ", df1["Combined"][1])

    x = df1[feat].values
    y = df["Label"].values

    if DictVect == False:

        dictVectorizer = DictVectorizer()

        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram, wordpiece_ngrams=wordpiece_gram, binary=False)
        x_dict = vectorizerWords.fit_transform(x)
        #print("first instance as features:", x_dict[0])
        x_train = dictVectorizer.fit_transform(x_dict)
        #print("Vocab size train: ", len(dictVectorizer.vocabulary_))

        if tfidf == True:

            tfIdfTransformer = TfidfTransformer(sublinear_tf=True)

            x_train = tfIdfTransformer.fit_transform(x_train)

            return x_train, y, dictVectorizer, tfIdfTransformer

        else:

            return x_train, y, dictVectorizer, tfIdfTransformer

    else:
        vectorizerWords = Featurizer(word_ngrams=word_gram, char_ngrams=char_gram, wordpiece_ngrams=wordpiece_gram)
        x_dict = vectorizerWords.fit_transform(x)
        x_test = DictVect.transform(x_dict)

        #print("Vocab size: ", len(DictVect.vocabulary_))

        if tfidf != False:

            x_test = tfIdfTransformer.transform(x_test)

            return x_test, y, DictVect, tfidf

        else:

            return x_test, y, DictVect, tfIdfTransformer



def train_eval(classifier, X_train, y_train, X_test, y_test, ensemble = False):

    """
    Adapted from https://github.com/bplank/bleaching-text/
    Classifier has been changed from LinearSVC to Logistic Regression
    """

    print()

    classifier.fit(X_train, y_train)

    y_predicted_test = classifier.predict(X_test)
    y_predicted_train = classifier.predict(X_train)

    accuracy_dev = accuracy_score(y_test, y_predicted_test)
    accuracy_train = accuracy_score(y_train, y_predicted_train)
    print("Classifier accuracy train: {0:.2f}".format(accuracy_train*100))

    print("===== dev set ====")
    #print("Classifier: {0:.3f}".format(accuracy_dev*100))
    print("Classifier: {0:.3f}".format(f1_score(y_test, y_predicted_test, average="weighted")*100))

    #print(classification_report(y_test, y_predicted_test, digits=4))

    if ensemble == True:
        return y_predicted_test

    return f1_score(y_test, y_predicted_test, average="weighted"), accuracy_score(y_test, y_predicted_test), y_predicted_test

def majorityVote(list1, list2, list3, y_test):
    votes = list()
    for i in range(len(list1)):
        class_0 = 0
        class_1 = 0

        if list1[i] == 0: class_0 += 1
        else: class_1 += 1

        if list2[i] == 0: class_0 += 1
        else: class_1 += 1

        if list3[i] == 0: class_0 += 1
        else: class_1 += 1

        if class_0 > class_1: votes.append(0)
        else: votes.append(1)
    
    print("F1: {}, Acc: {}".format(f1_score(y_test, votes, average="weighted"), accuracy_score(y_test, votes)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wordN", default="1", type=str, help="Log dir name")
    parser.add_argument("--charN", default="0", type=str, help="Log dir name")
    parser.add_argument("--wpieceN", default="0", type=str, help="Log dir name")
    args = parser.parse_args()

    #print(os.listdir("data/"))

    train_data_path = "data/train_lower_entities.tsv"
    #train_data_path = "data/train.tsv"
    test_data_path = "data/valid_lower_entities.tsv"
    #test_data_path = "data/valid.tsv"
    wordN = args.wordN
    charN = args.charN
    wordpieceN = args.wpieceN
    use_tfidf = True

    classifier = LinearSVC(penalty='l2', loss='squared_hinge', \
                           dual=True, tol=0.0001, C=1.0, multi_class='ovr', \
                           fit_intercept=True, intercept_scaling=1, class_weight=None)

    # X_train, y_train, dictvect, tfidfvect = load_file(train_data_path, feat = "Text", tfidf=use_tfidf, tfIdfTransformer=None, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)
    # X_dev, y_dev, _, _ = load_file(test_data_path, feat = "Text", DictVect=dictvect, tfidf=use_tfidf, tfIdfTransformer=tfidfvect, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)

    # X, y = load2Files("data/train.tsv", "data/valid.tsv")

#    kfold(X, y)

    # f1_test, acc_test, _ = train_eval(classifier, X_train, y_train, X_dev, y_dev)
    # print("weighted f1: {0:.5f}".format(f1_test))
    # print("accuracy: {0:.5f}".format(acc_test))

    # Ensemble
    X_train, y_train, dictvect, tfidfvect = load_file(train_data_path, feat = "Text", tfidf=use_tfidf, tfIdfTransformer=None, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)
    X_dev, y_dev, _, _ = load_file(test_data_path, feat = "Text", DictVect=dictvect, tfidf=use_tfidf, tfIdfTransformer=tfidfvect, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)
    pred1 = train_eval(classifier, X_train, y_train, X_dev, y_dev, ensemble = True)

    print('1 done')

    X_train, y_train, dictvect, tfidfvect = load_file(train_data_path, feat = "Entities", tfidf=use_tfidf, tfIdfTransformer=None, word_gram='1-4')
    X_dev, y_dev, _, _ = load_file(test_data_path, feat = "Entities", DictVect=dictvect, tfidf=use_tfidf, tfIdfTransformer=tfidfvect, word_gram='1-4')
    pred2 = train_eval(classifier, X_train, y_train, X_dev, y_dev, ensemble = True)

    print('2 done')

    X_train, y_train, dictvect, tfidfvect = load_file(train_data_path, feat = "Entities_Details", tfidf=use_tfidf, tfIdfTransformer=None, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)
    X_dev, y_dev, _, _ = load_file(test_data_path, feat = "Entities_Details", DictVect=dictvect, tfidf=use_tfidf, tfIdfTransformer=tfidfvect, word_gram=wordN, char_gram=charN, wordpiece_gram=wordpieceN)
    pred3 = train_eval(classifier, X_train, y_train, X_dev, y_dev, ensemble = True)

    print('3 done')

    majorityVote(pred1, pred2, pred3, y_dev)