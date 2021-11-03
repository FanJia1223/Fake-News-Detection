
# coding: utf-8

# In[ ]:


import os
import re
import json
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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


# In[ ]:


claim = []
index = []
# Write the labels and corresponding claims into a list
for c in claims: 
    claim.append(c['claim'])
for c in related_sentences: 
    index.append(c)


# In[ ]:


# Perform simple cleaning to the related sentences
def cleaning(article):
    
    tokenizer = RegexpTokenizer(r'\w+')
    lrStem = LancasterStemmer()
    wnLemm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    article.replace('-',' ')
    
    article_words = tokenizer.tokenize(article)
    #article_words = [lrStem.stem(w) for w in article_words] # add stemmer
    article_words = [wnLemm.lemmatize(w,'v') for w in article_words] # add lemmatizer           
    article_words = [w.lower() for w in article_words if not w in stop_words] # remove stop word

    return article_words


# In[ ]:
#print(related_sentences)

final_features = []
dim = 20

# First loop across indices which represents different claims
for ind in index: 
    print("claim: ", ind)
    splitted_sentences = []
    vocab = []
    sentences = []
    cleaned = []
    features = []
    final = []

  # Apply the cleaning function to sentences and corresponding claim
    for sen in related_sentences[ind]:
        sentences = cleaning(sen)
        splitted_sentences.append(sentences)
  
    cleaned = cleaning(claim[int(ind)])
    splitted_sentences.append(cleaned)
  
  # Create word embedding model for the splitted sentence
    model = Word2Vec(splitted_sentences, size=dim, window=5, min_count=1)

  # Obtain vocabulary list
    vocab = list(model.wv.vocab)
    length = len(vocab)
    avg_vectors = []

  # Record the vector representing each word in sentence and calculate the average across such sentence
    for i in range(0, len(splitted_sentences)):
        vec_sum = [0] * dim
        vec_count = 0
        for word in splitted_sentences[i]: 
            if word in vocab: 
                vector = list(model[str(word)])
                vec_sum = [sum(i) for i in zip(vec_sum, vector)]
                vec_count += 1
            else: 
                continue
        if vec_count > 0: 
            avg_vector = [x / vec_count for x in vec_sum]
            features.append(avg_vector)
        else: 
            features.append(vec_sum)

    if len(splitted_sentences) < 6:
        diff = 6 - len(splitted_sentences)
        for m in range(0, diff): 
            fillin = [0] * dim
            features.append(fillin)
  # Combine all vectors under the same claim together
    final = np.asarray(features).flatten()
  # Attach the resulting overall vector in the final list as features
    final_features.append(final.tolist())


# In[ ]:


X = pd.DataFrame(final_features)


# In[ ]:


from joblib import dump, load
model = load('lrmodel.joblib')


# In[ ]:


prediction = model.predict(X)


# In[ ]:


with open(PREDICTIONS_FILEPATH, 'a') as predictions:
    for p, i in zip(prediction,index):
        predictions.write(str(i)+','+str(p) + '\n')

