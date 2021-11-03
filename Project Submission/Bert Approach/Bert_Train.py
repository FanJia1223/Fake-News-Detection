#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pytorch-pretrained-bert pytorch-nlp
#!pip install numpy
# BERT imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


# In[2]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def fromat_and_tokenize(queries):
    sentences = ["[CLS] " + query + " [SEP]" for query in queries]
  # Tokenize with BERT tokenizer
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    return tokenized_texts; 

def pad_sentences(texts,max_length):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in texts], maxlen=max_length, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attention_masks(input_ids):
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks
def setup_dataloader(inputs,labels,batch_size):
    sentences = fromat_and_tokenize(inputs)
    input_ids = pad_sentences(sentences,MAX_LEN)
    attention_masks = create_attention_masks(input_ids)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
 
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[3]:


import json
with open('related_sentences.json', 'r') as f:
    related_sentences = json.load(f)
with open('train.json', 'r') as f:
    claims = json.load(f)


# In[10]:


labels = []
inputs = []
for claim, index in zip(claims,related_sentences):
    concat_sentence = claim['claim']
    for sentence in related_sentences[index]:
        concat_sentence = concat_sentence + " " + sentence
    if(len(concat_sentence) > 512):
        concat_sentence = concat_sentence[:512]
    inputs.append(concat_sentence)
    labels.append(claim['label'])


# In[11]:


MAX_LEN = 512
sentences = fromat_and_tokenize(inputs)
input_ids = pad_sentences(sentences,MAX_LEN)
attention_masks = create_attention_masks(input_ids)


# In[12]:


# Use train_test_split to split our data into train and validation sets for training
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. 
batch_size = 8
# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# In[13]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.cuda()


# In[14]:


# BERT fine-tuning parameters

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5,warmup=.1)
# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Store our loss and accuracy for plotting
train_loss_set = []
# Number of training epochs 
epochs = 20
current_epoch = 1
# BERT training loop
for _ in trange(epochs, desc="Epoch"):  
    ## TRAINING
    # Set our model to training mode
    model.train()  
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    train_loss_set_epoch = []
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())
        train_loss_set_epoch.append(loss.item()) 
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # plot training performance
    plt.figure(figsize=(15,8))
    plt.title("Training loss for epoch " + str(current_epoch))
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set_epoch)
    plt.savefig('train_loss_' + str(current_epoch) + '.png')
    current_epoch = current_epoch + 1
        
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.savefig('train_loss_overall.png')
 

# In[21]:
torch.save(model.state_dict,'states/checkpoint.pt.tar')
torch.save(model,'states/model.pt.tar')

