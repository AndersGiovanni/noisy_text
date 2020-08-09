"""
Sentence classification.
Based on https://github.com/huggingface/transformers
"""

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER

import random
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tensorflow as tf
import torch as t
from transformers import *

from classify import encode_label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    pretrained_weights = "bert-base-uncased"
    #model = BertModel.from_pretrained(pretrained_weights)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = t.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
