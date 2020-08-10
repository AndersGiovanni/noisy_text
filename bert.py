"""
Sentence classification.
Based on https://github.com/huggingface/transformers and https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
"""

from myutils import Featurizer, EmbedsFeaturizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM, TWEET_DELIMITER

import random
import os
import argparse
import time
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from tqdm import tqdm

from classify import encode_label

# Fix seed for replicability
seed=103
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def loadFile(file):
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

    X = torch.tensor(input_ids)
    y = torch.tensor(y)
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

def f1(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. 
    Reference: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    MAX_LEN = 64

    train_data_path = "data/train_lower_entities.tsv"
    test_data_path = "data/valid_lower_entities.tsv"

    X_train, y_train, mask_train = loadFile(train_data_path)
    X_test, y_test, mask_test = loadFile(test_data_path)

    train = makeDataLoader(X_train, y_train, mask_train)
    test = makeDataLoader(X_test, y_test, mask_test)

    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
    #                                                        num_labels = 2,
    #                                                        output_attentions = False,
    #                                                        output_hidden_states = False)

    model = BertForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert',
                                                        num_labels = 2,
                                                        output_attentions = False,
                                                        output_hidden_states = False)

    #model = AutoModel.from_pretrained()

    optimizer = AdamW(model.parameters(),
                        lr = 2e-5,
                        eps = 1e-8)

    epochs = 2

    total_steps = len(train) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    loss_values = list()

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train), elapsed))

            # Unpack this training batch from our dataloader. 
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to('cpu')
            b_input_mask = batch[1].to('cpu')
            b_labels = batch[2].to('cpu')

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in test:
            
            # Add batch to cpu
            batch = tuple(t.to("cpu") for t in batch)
            
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
            tmp_eval_f1 = f1(logits, label_ids)                     # f1
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  F1: {0:.2f}".format(eval_f1/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")