"""
Making predictions using a bert transformer model. This model is a covid-twitter-bert model fine-tuned on
on training data from WNUT. 

"""

import random
import argparse
import time
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm

seed=103
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def encode_label(label):
    """
    Convert UNINFORMATIVE to 0 and INFORMATIVE to 1
    """
    if label == "UNINFORMATIVE": return 0
    else: return 1

def loadFile(file, device):
    """
    Load file and apply preprocessing for BERT model
    """
    df = pd.read_csv(file, sep='\t')
    df.Label = df.Label.apply(lambda x: encode_label(x))

    X = df.Text
    y = df.Label

    # Define tokenizer
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')

    # Encode sentences to ids
    input_ids = list()
    for sent in tqdm(X):
        encoded_sent = tokenizer.encode(sent, 
                                        add_special_tokens = True,
                                        truncation = True,
                                        max_length = 128) 
                                        #return_tensors = 'pt')

        input_ids.append(encoded_sent)

    # Pad/truncate sentences
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids,
                                                                maxlen=128,
                                                                dtype='long',
                                                                value=0,
                                                                truncating='post',
                                                                padding='post')

    # Attention Masks
    attention_masks = list()
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    X = torch.tensor(input_ids).to(device)
    y = torch.tensor(y).to(device)
    attention_masks = torch.tensor(attention_masks)

    return X, y, attention_masks

def makeDataLoader(X, y, attention_masks):
    """
    Make PyTorch iterator
    """
    batch_size = 16

    data = TensorDataset(X, attention_masks, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def findDevice():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def modelEval(model, data, device):
    """
    Making prediction on test data. 
    Printing:
        - Accuracy
        - Weighted F1
    """

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in data:
        
        # Add batch to cpu
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)   # acc
        tmp_eval_f1 = f1_score(np.argmax(logits, axis = 1).flatten(), label_ids.flatten(), average="weighted")                     # f1
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        eval_f1 += tmp_eval_f1

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.5f}".format(eval_accuracy/nb_eval_steps))
    print("  F1: {0:.5f}".format(eval_f1/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

if __name__ == "__main__":
    
    device = findDevice()

    train_data_path = "data/train_lower_entities.tsv"
    test_data_path = "data/valid_lower_entities.tsv"

    X_test, y_test, mask_test = loadFile(test_data_path, device)

    test = makeDataLoader(X_test, y_test, mask_test)

    model = AutoModelForSequenceClassification.from_pretrained('models/covid-bert-fine-tuned1').to(device)

    modelEval(model, test, device)