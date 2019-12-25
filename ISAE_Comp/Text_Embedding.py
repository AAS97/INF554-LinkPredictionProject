#!/usr/bin/env python
# coding: utf-8

# # Text Embedding
# ## Discover Node Information

# In[4]:


#get_ipython().system('pip install nltk')


# In[6]:


#Getting Stopwords
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Download pre-generated list of common words and Stemmer
stop = stopwords.words('french')
sno = SnowballStemmer('french')
print(sno.stem('aimer'))
print (stop)


# In[28]:


import os

directory = os.fsencode('./data/node_information/text')
node_info = {}


#for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".txt") :
#         with open('./data/node_information/text/'+filename, 'r', encoding='utf-8', errors='ignore') as cur_file:
#            node_info[filename[:-4]] = ''' '''
#            for line in cur_file:
#                data = line.replace('\n', '')
#                node_info[filename[:-4]] += data


# In[ ]:


from joblib import Parallel, delayed
import multiprocessing


node_info = {}
def processFile(file):
    filename = os.fsdecode(file)
    if filename.endswith(".txt") and int(filename[:-4])<1000 :
         with open('./data/node_information/text/'+filename, 'r', encoding='utf-8', errors='ignore') as cur_file:
            node_info[filename[:-4]] = ''' '''
            for line in cur_file:
                data = line.replace('\n', '')
                node_info[filename[:-4]] += data

num_cores = multiprocessing.cpu_count()
print(num_cores)

Parallel(n_jobs = -2, require='sharedmem')(delayed(processFile)(file) for file in os.listdir(directory))


# In[ ]:


#print(node_info['3136'])
import json
with open('./data/node_information/node_info.json', 'w') as file:
    json.dump(node_info, file)


# In[ ]:


tokenizer = nltk.RegexpTokenizer(r'\w+')
node_info_tokenized = {}
for node in node_info:
    node_info_tokenized[node] = tokenizer.tokenize(node_info[node])
    
with open('./data/node_information/node_info_token.json', 'w') as file:
    json.dump(node_info_tokenized, file)

