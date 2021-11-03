#!/usr/bin/env python
# coding: utf-8

# In[34]:

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
import nltk
from nltk.stem import LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

import os
import re
import json
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path

import random
random.seed()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning) 

#get_ipython().run_line_magic('matplotlib', 'inline')
# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)


# In[ ]:


# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
ARTICLES_FILEPATH = '/usr/local/dataset/articles'
# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'

# Read in the metadata file.
with open(METADATA_FILEPATH, 'r') as f:
    claims = json.load(f)


# In[ ]:


# extract all file paths
all_files = [pth for pth in Path(ARTICLES_FILEPATH).glob("**/*") if pth.is_file() and not pth.name.startswith(".")]


# In[ ]:


# input: a list of files, output: a dictionary of articles, keys are article id, values are article string
def read_articles(file_list):
    all_articles = {}
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            filename = os.path.basename(file_path)
            filename = filename.replace(".txt","")
            file_data = file.read()
            all_articles[filename] = file_data
    return all_articles


# In[ ]:


# save all articles in a dictionary
all_articles = read_articles(all_files)


# In[ ]:


# input: string of article content, output: list of lines in article
def tosentences(article):
    sentence_list = sent_tokenize(article)
    sentence_list = list(filter(None,sentence_list))

    return sentence_list


# In[ ]:


# input: string of article content, output: list of cleaned words
def cleandata(article):
    tokenizer = RegexpTokenizer(r'\w+')
    lrStem = LancasterStemmer()
    wnLemm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
  
    article = article.replace("\n"," ").replace("\\","")
    article_words = tokenizer.tokenize(article)
    article_words = [w.lower() for w in article_words] # add lowercase 
    article_words = [wnLemm.lemmatize(w,'v') for w in article_words] # add lemmatizer           
    article_words = [w for w in article_words if not w in stop_words] # remove stop word

    return article_words


# In[ ]:


# input: string of claim sentence & list of sentences from article, output: tfidf dataframe
def tfidfsentence(article_list):
    claim = article_list[0]
    features = list(dict.fromkeys(claim))
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(article_list)
    X = X.toarray()
    tfidfdf = pd.DataFrame(np.round(X,2),columns=vectorizer.get_feature_names())
  
    return tfidfdf


# In[ ]:


# new cossim function, return the list of 5 top related sentences with highest cosine similarities
def top5cossim(article_list):
    senlist=[]
    cosines = []
    df_tfidf = tfidfsentence(article_list)
    claim = df_tfidf.iloc[0]

    for index in range(1,len(article_list)):
        sentvec = df_tfidf.iloc[index].values
        cosines.append(cosine_similarity([sentvec],[claim]))
    cosines = np.concatenate(np.concatenate(cosines, axis=0), axis=0)
    cosines=pd.DataFrame(cosines)
    cosines.columns=['cosine']
  
    top5index = list(cosines.iloc[cosines.cosine.argsort()[::-1][:5]].index+1) # plus one because of the claim at index 0
    for a in top5index:
        senlist.append(article_list[a])

    return senlist


# In[ ]:


# save top 5 most related sentences from related articles into a dictionary for each claims
related_sentences = dict.fromkeys(range(len(claims)), []) 
for ii in range(len(claims)): # change to range(len(claims))
    articles = claims[ii]["related_articles"]
    allrelated = ''
    for articleid in articles:
        allrelated=allrelated+'\n'+all_articles[str(articleid)] # read all the related articles

    claim = claims[ii]["claim"]
    allrelated = allrelated.replace("\n"," ").replace(":",".").replace(";",".")
    sentence_list=tosentences(allrelated)
    sentence_list.insert(0, claim)  # insert the claim to index 0

    related_sentences[ii]=top5cossim(sentence_list) # append the list of related sentences into the dictionary


# In[17]:


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
    input_ids = pad_sentences(sentences,512)
    attention_masks = create_attention_masks(input_ids)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
 
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader 
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[18]:


labels = []
inputs = []
ids = []
for claim, index in zip(claims,related_sentences):
    concat_sentence = claim['claim']
    for sentence in related_sentences[index]:
        concat_sentence = concat_sentence + " " + sentence
    if(len(concat_sentence) > 512):   
        concat_sentence = concat_sentence[:512]
    inputs.append(concat_sentence)
    labels.append(claim['label'])
    ids.append(claim['id'])


# In[20]:

if(n_gpu > 0):
    model = torch.load('/usr/src/app/model.pt.tar')
else:
    model = torch.load('/usr/src/app/model.pt.tar',map_location='cpu')


# In[21]:
model.eval()


# In[22]:
if(n_gpu > 0):
    model.cuda()


# In[23]:


dataloader = setup_dataloader(inputs,labels,4)


# In[24]:

predictions , true_labels = [], []
# Predict 
for batch in dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
    # Forward pass, calculate logit predictions
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

# Import and evaluate each test batch using Matthew's correlation coefficient
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

#print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))

# In[33]:

PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'
with open(PREDICTIONS_FILEPATH, 'a') as predictions:
    for p, i in zip(flat_predictions,ids):
        predictions.write(str(i)+','+str(p) + '\n')

# In[ ]:




