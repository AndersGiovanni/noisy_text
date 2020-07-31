__author__ = "bplank"
# Adapted from https://github.com/bplank/bleaching-text/

import nltk
from sklearn.base import TransformerMixin
import numpy as np
import json
from transformers import BertTokenizer

PREFIX_WORD_NGRAM="W:"
PREFIX_CHAR_NGRAM="C:"
PREFIX_WORDPIECE_NGRAM="WP:"
TWEET_DELIMITER = " NEWLINE "


def get_size_tuple(ngram_str):
    """
    Convert n-gram string to tuple
    :param ngram_str:  "1-3" (lower and upper bound separated by hyphen)
    :return: tuple
    >>> get_size_tuple("3-5")
    (3, 5)
    >>> get_size_tuple("1")
    (1, 1)
    """
    if "-" in ngram_str:
        lower, upper = ngram_str.split("-")
        lower = int(lower)
        upper = int(upper)
    else:
        lower = int(ngram_str)
        upper = lower
    return (lower, upper)

class EmbedsFeaturizer(TransformerMixin):
    """ our own featurizer for embedding features """
    def __init__(self, embeds, only_mean=False):
        self.emb = embeds
        self.only_mean = only_mean # only average embedding if active

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        out= [self._emb_feats(tweets) for tweets in X]
        return out

    def _emb_feats(self, tweets):
        d={}
        tweets = tweets.split(TWEET_DELIMITER)
        word_vec = []
        for tweet in tweets:
            words = tweet.split(" ")  # trivial tokenization
            for w in words:
                if w in self.emb:
                    word_vec.append(self.emb.get(w))
                if w.lower() in self.emb:
                    word_vec.append(self.emb.get(w.lower()))
                # ignore _UNKs 
            if len(word_vec) == 0:
#                print(tweet)
                continue # skip tweet too short (notice: we also do not tokenize thus lower coverage..)

        if len(word_vec) == 0:
            print(tweets)
        # for now just join all tweets together
        avg_emb = np.mean(word_vec, axis=0)
        sd_emb = np.std(word_vec, axis=0)
        sum_emb = np.mean(word_vec, axis=0)
        
        if self.only_mean:
            for i, val in enumerate(avg_emb):
                d["d_{}_{}".format(i, "mean")] = val
        else:
            for f, vec in (("mean", avg_emb), ("std", sd_emb), ("sum", sum_emb)):
                for i, val in enumerate(vec):
                    d["d_{}_{}".format(i, f)] = val
                        
            d["overall_max"] = np.max(word_vec)
            d["overall_min"] = np.min(word_vec)
            d["emb_cov_rate"] = np.sum([1 for w in words if w in self.emb])/len(words)
        return d

class Featurizer(TransformerMixin):
    """Our own featurizer: extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        for all tweets of a user
        """
        out= [self._ngrams(tweets) for tweets in X]
        return out

    def __init__(self,word_ngrams="1",char_ngrams="0",wordpiece_ngrams="0",binary=True,rm_user_url=False):
        """
        binary: whether to use 1/0 values or counts
        lowercase: convert text to lowercase
        remove_stopwords: True/False
        """
        self.data = [] # will hold data (list of dictionaries, one for every instance)
        self.binary=binary
        self.word_ngram_size = get_size_tuple(word_ngrams)
        self.char_ngram_size = get_size_tuple(char_ngrams)
        self.wordpiece_ngram_size = get_size_tuple(wordpiece_ngrams)

        self.rm_user_url=rm_user_url
        if self.wordpiece_ngram_size != '0':
            self.wordpiece_tokenizer = BertTokenizer('ml-bert.vocab.txt', do_lower_case=False)


    def _ngrams(self,tweets):
        """
        extracts word or char n-grams

        range defines lower and upper n-gram size

        >>> f=Featurizer(word_ngrams="1-3")
        >>> d = f._ngrams("this is a test")
        >>> len(d)
        9
        >>> f=Featurizer(word_ngrams="0", char_ngrams="2-4")
        >>> d2 = f._ngrams("this")
        >>> len(d2)
        6
        """

        d={} # new dictionary that holds features for current instance

        tweets = tweets.split(TWEET_DELIMITER)

        lower, upper = self.word_ngram_size
        if lower != 0:
            for n in range(lower,upper+1):
                for tweet in tweets:
                    if self.rm_user_url:
                        tweet=tweet.replace("USER","")
                        tweet=tweet.replace("URL","")

                    ## word n-grams
                    for gram in nltk.ngrams(tweet.split(" "), n):
                        gram = "{}_{}".format(PREFIX_WORD_NGRAM, "_".join(gram))
                        if self.binary:
                            d[gram] = 1 #binary
                        else:
                            d[gram] = d.get(gram,0)+1

        c_lower, c_upper = self.char_ngram_size
        if c_lower != 0:
            for n in range(c_lower, c_upper + 1):
                for tweet in tweets:
                    if self.rm_user_url:
                        tweet = tweet.replace("USER", "")
                        tweet = tweet.replace("URL", "")

                    ## char n-grams
                    for gram in nltk.ngrams(tweet, n):
                        gram = "{}_{}".format(PREFIX_CHAR_NGRAM, "_".join(gram))
                        if self.binary:
                            d[gram] = 1  # binary
                        else:
                            d[gram] = d.get(gram, 0) + 1

        wp_lower, wp_upper = self.wordpiece_ngram_size
        if wp_lower != 0:
            # Rob: I swapped the order of 'n in range' and 'tweet in tweets'
            # for efficiency reasons, now we only have to tokenize once
            for tweet in tweets:
                tok = self.wordpiece_tokenizer.tokenize(tweet)
                if self.rm_user_url:
                    tweet = tweet.replace("USER", "")
                    tweet = tweet.replace("URL", "")

                for n in range(wp_lower, wp_upper + 1):
                    ## wordpiece n-grams
                    for gram in nltk.ngrams(tok, n):
                        gram = "{}_{}".format(PREFIX_WORDPIECE_NGRAM, "_".join(gram))
                        if self.binary:
                            d[gram] = 1 #binary
                        else:
                            d[gram] = d.get(gram,0)+1

        return d


if __name__ == "__main__":
    import doctest
    doctest.testmod()
