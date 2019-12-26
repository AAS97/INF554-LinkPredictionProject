#!/usr/bin/env python
# coding: utf-8

# # Text Embedding
# ## Discover Node Information
# Install Natural Language Processing ToolKit

# In[2]:


#get_ipython().system('pip3 install nltk')


# ## Getting Stopwords

# In[62]:


import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import json
import matplotlib.pyplot as plt
import numpy as np


# Download pre-generated list of common words and words root dictionnary

# Let import the data in a dict to access it

# In[42]:


from joblib import Parallel, delayed
import multiprocessing
import os

#set path to the data directory
directory = os.fsencode('./data/node_information/text')

#Highest index of node we want to precess
max_node = 1000

node_info = {}


# In[28]:


# Sequential computation if wanted but longer

#for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".txt") :
#         with open('./data/node_information/text/'+filename, 'r', encoding='utf-8', errors='ignore') as cur_file:
#            node_info[filename[:-4]] = ''' '''
#            for line in cur_file:
#                data = line.replace('\n', '')
#                node_info[filename[:-4]] += data


# In[ ]:


def processFile(file):
    ### write an entry on node_info as node_nb : text where node info is the title of the doc
        
    filename = os.fsdecode(file)
    if filename.endswith(".txt") and int(filename[:-4]) < max_node :
         with open('./data/node_information/text/'+filename, 'r', encoding='utf-8', errors='ignore') as cur_file:
            node_info[filename[:-4]] = ''' '''
            for line in cur_file:
                data = line.replace('\n', '')
                node_info[filename[:-4]] += data


# In[ ]:


#Get number of available CPU on computer
num_cores = multiprocessing.cpu_count()
print("%s cores available, going to use %s for parallel computation" %[num_cores, num_cores-1])
#Run parallel computation with joblib
Parallel(n_jobs = -2, require='sharedmem')(delayed(processFile)(file) for file in os.listdir(directory))


# In[ ]:


# Print node_info to json
with open('./data/node_information/node_info.json', 'w') as file:
    json.dump(node_info, file)


# In[53]:


tokenizer = nltk.RegexpTokenizer(r'\w+')
# ew dict for tokenized form
node_info_tokenized = {}


# In[ ]:


def tokenize_dict(node):
    node_info_tokenized[node] = tokenizer.tokenize(node_info[node])
    
Parallel(n_jobs = -2, require='sharedmem')(delayed(tokenize_dict)(node) for node in node_info)

with open('./data/node_information/node_info_token.json', 'w') as file:
    json.dump(node_info_tokenized, file)


# ## Bag of words
# Text embedding as a dictionnary and occurence count
# https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

# In[3]:


# load data from file


# In[17]:


with open('./data/computed/node_info_token.json', 'r') as f:
    node_info_tokenized = json.load(f)
node_info_tokenized = {int(k):v for k,v in node_info_tokenized.items()}
print(node_info_tokenized[1])

with open('./data/computed/node_info.json', 'r') as f:
    node_info = json.load(f)
node_info = {int(k):v for k,v in node_info.items()}
print(node_info[1])


# In[40]:


## Bag of word for each node or for all the nodes at once ?
from nltk.probability import FreqDist

#counting word frequency in text
""" fdist = FreqDist(node_info_tokenized[1])
fdist2 = FreqDist(node_info_tokenized[2])

print(fdist)


# Use bag of word to plot most commin words, fix a limit of useless common words and remove them from bag of words

# In[37]:


fdist.plot(30,cumulative=False)
fdist2.plot(30,cumulative=False)
plt.show() """


# Remark : None of the 30 most recurent words give context to the page. Need to remove stop words

# In[58]:


stop_words = stopwords.words('french')
print("10 most used french words")
print(stop_words[0:10])


# In[43]:


node_info_filtered = {}

def remove_stopwords(node):
    node_info_filtered[node] = []
    for w in node_info_tokenized[node]:
        if w not in stop_words:
            node_info_filtered[node].append(w)
print("Begin remove stopwords computation")
Parallel(n_jobs = -2, require='sharedmem')(delayed(remove_stopwords)(node) for node in node_info_tokenized)

#remove_stopwords(1)


# In[ ]:


with open('./data/node_information/node_info_filtereder.json', 'w') as file:
    json.dump(node_info_filtered, file)


# In[29]:


fdist = FreqDist(node_info_filtered[1])
fdist.plot(30,cumulative=False)
plt.show()


# Remark : we now see that some words appear multiple times with differents typo. Need to Normalize the lexicon (Lemmization and Stemming)

# In[44]:


sno = SnowballStemmer('french')
print("Linguistic root of \'aimer\'")
print(sno.stem('aimer'))

node_info_snl = {}

def stemming_lemming(node):
    node_info_snl[node] = []
    for w in node_info_filtered[node]:
        node_info_snl[node].append(sno.stem(w))

print("Begin stemming computation")
Parallel(n_jobs = -2, require='sharedmem')(delayed(stemming_lemming)(node) for node in node_info_filtered)

#stemming_lemming(1)


# In[ ]:


with open('./data/node_information/node_info_snl.json', 'w') as file:
    json.dump(node_info_snl, file)


# In[47]:


fdist = FreqDist(node_info_snl[999])
fdist.plot(30,cumulative=False)
plt.show()


# In[100]:


#cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
#text_counts= cv.fit_transform(data['Phrase'])

import pandas as pd 

print("Begin Dict computation")
dict_full = []
def build_voca(node):
    for w in node_info_snl[node]:
        if w not in dict_full:
            dict_full.append(w)
#build_voca(1)          
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_voca)(node) for node in node_info_snl)


# In[86]:


with open('./data/node_information/dict.json', 'w') as file:
    json.dump(dict_full, file)


# In[87]:


#print(dict_full)


# In[99]:

print("Begin BOW computation")
bow = {}
def build_bag(node):
    bow[node] = np.zeros(shape=len(dict_full), dtype = int)
    for w in node_info_snl[node]:
        bow[int(node)][dict_full.index(w)] += 1            
    bow[node] = bow[node].tolist()
#build_bag(1)
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_bag)(node) for node in node_info_snl)


# In[98]:





# In[91]:


with open('./data/node_information/BOW.json', 'w') as file:
    json.dump(bow, file)


# In[13]:


##print BOW to json for using in the ML algo


# ## Word2Vec
# 
# Embbed the word 

