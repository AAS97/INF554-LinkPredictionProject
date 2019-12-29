'''
Import libraries
'''

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

'''
For parallel computing
'''
from joblib import Parallel, delayed
import multiprocessing
import os

print("{0} cores available, going to use {1} for parallel computation".format(multiprocessing.cpu_count(), multiprocessing.cpu_count()-1), flush=True)

'''
    Load text data in appropriate format
'''
#set path to the data directory where text files are
directory = os.fsencode('./data/node_information/text')

node_info = {}
def processFile(file, max_node = 10):
    '''
        write an entry on node_info as node_nb : text where node info is the title of the doc
    '''    
    filename = os.fsdecode(file)
    if filename.endswith(".txt") and int(filename[:-4]) < max_node :
        #, encoding='utf-8', errors='ignore'
         with open('./data/node_information/text/'+filename, 'r') as cur_file:
            node_info[filename[:-4]] = ''' '''
            for line in cur_file:
                node_info[filename[:-4]] += line
            node_info[filename[:-4]] = node_info[filename[:-4]].replace('\n', '')

#Run parallel loading of data
print("Starting data loading", flush=True)
Parallel(n_jobs = -2, require='sharedmem')(delayed(processFile)(file) for file in os.listdir(directory))
#making sure keys are integers
node_info = {int(k):v for k,v in node_info.items()}

# Print node_info to json
with open('./ISAE_Comp/out/node_info.json', 'w') as file:
    json.dump(node_info, file)

print("Finished loading {0} text data to dictionnary and saved it to file".format(len(node_info.keys())), flush=True)


'''
    Tokenize each text
'''
#define regex for tokenization
tokenizer = nltk.RegexpTokenizer(r'\w+')

node_info_tokenized = {}
def tokenize_dict(node):
    '''
        add an entry on node_info_tokenized dict for the node as word list
    '''
    node_info_tokenized[node] = tokenizer.tokenize(node_info[node])

print("Starting tokenization", flush=True)  
Parallel(n_jobs = -2, require='sharedmem')(delayed(tokenize_dict)(node) for node in node_info)

#making sure keys are integers
node_info_tokenized = {int(k):v for k,v in node_info_tokenized.items()}

with open('./ISAE_Comp/out/node_info_token.json', 'w') as file:
    json.dump(node_info_tokenized, file)
print("Finished tokenizing {0} entries to dictionnary and saved it to file".format(len(node_info_tokenized.keys())), flush=True)




'''
    Removing stopwords
'''
print("Downloading french stopwords", flush=True)
nltk.download('stopwords')
stop_words = stopwords.words('french')


node_info_filtered = {}
def remove_stopwords(node):
    '''
        add an entry on node_info_filtered dict for the node as word list removing stopwords from node_info_tokenized
    '''
    node_info_filtered[node] = []
    for w in node_info_tokenized[node]:
        if w not in stop_words:
            node_info_filtered[node].append(w)

print("Starting stopword removal", flush=True)  
Parallel(n_jobs = -2, require='sharedmem')(delayed(remove_stopwords)(node) for node in node_info_tokenized)

with open('./ISAE_Comp/out/node_info_filtered.json', 'w') as file:
    json.dump(node_info_filtered, file)
print("Finished cleaning {0} entries of dictionnary and saved it to file".format(len(node_info_filtered.keys())), flush=True)


'''
    Stemming and lemming : word normalization
'''
print("Downloading french normalization tool", flush=True)
stemmer = FrenchStemmer()

node_info_snl = {}
def stemming_lemming(node):
    '''
        add an entry on node_info_snl dict for the node lemming and stemming all words of node_info_filtered
    '''
    node_info_snl[node] = []
    for w in node_info_filtered[node]:
        node_info_snl[node].append(stemmer.stem(w))

print("Starting stemming & lemming", flush=True)    
Parallel(n_jobs = -2, require='sharedmem')(delayed(stemming_lemming)(node) for node in node_info_filtered)

with open('./ISAE_Comp/out/node_info_snl.json', 'w') as file:
    json.dump(node_info_snl, file)
print("Finished s&l on {0} entries of dictionnary and saved it to file".format(len(node_info_snl.keys())), flush=True)

print('Program finished as expected', flush=True)