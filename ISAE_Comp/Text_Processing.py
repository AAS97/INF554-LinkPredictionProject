'''
Import libraries
'''

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
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

print("{0} cores available, going to use {1} for parallel computation".format(multiprocessing.cpu_count(), multiprocessing.cpu_count()-1))

'''
    Load text data in appropriate format
'''
#set path to the data directory where text files are
directory = os.fsencode('./data/node_information/text')

node_info = {}
def processFile(file, max_node = 2):
    '''
        write an entry on node_info as node_nb : text where node info is the title of the doc
    '''    
    filename = os.fsdecode(file)
    if filename.endswith(".txt") and int(filename[:-4]) < max_node :
         with open('./data/node_information/text/'+filename, 'r', encoding='utf-8', errors='ignore') as cur_file:
            node_info[filename[:-4]] = ''' '''
            for line in cur_file:
                data = line.replace('\n', '')
                node_info[filename[:-4]] += data

#Run parallel loading of data
print("Starting data loading")
Parallel(n_jobs = -2, require='sharedmem')(delayed(processFile)(file) for file in os.listdir(directory))

#making sure keys are integers
node_info = {int(k):v for k,v in node_info.items()}

# Print node_info to json
with open('./ISAE_Comp/out/node_info.json', 'w') as file:
    json.dump(node_info, file)

print("Finished loading {0} text data to dictionnary and saved it to file".format(len(node_info.keys())))


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

print("Starting tokenization")  
Parallel(n_jobs = -2, require='sharedmem')(delayed(tokenize_dict)(node) for node in node_info)

#making sure keys are integers
node_info_tokenized = {int(k):v for k,v in node_info_tokenized.items()}

with open('./ISAE_Comp/out/node_info_token.json', 'w') as file:
    json.dump(node_info_tokenized, file)
print("Finished tokenizing {0} entries to dictionnary and saved it to file".format(len(node_info_tokenized.keys())))




'''
    Removing stopwords
'''
print("Downloading french stopwords")
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

print("Starting stopword removal")  
Parallel(n_jobs = -2, require='sharedmem')(delayed(remove_stopwords)(node) for node in node_info_tokenized)

with open('./ISAE_Comp/out/node_info_filtered.json', 'w') as file:
    json.dump(node_info_filtered, file)
print("Finished cleaning {0} entries of dictionnary and saved it to file".format(len(node_info_filtered.keys())))


'''
    Stemming and lemming : word normalization
'''
print("Downloading french normalization tool")
sno = SnowballStemmer('french')

node_info_snl = {}
def stemming_lemming(node):
    '''
        add an entry on node_info_snl dict for the node lemming and stemming all words of node_info_filtered
    '''
    node_info_snl[node] = []
    for w in node_info_filtered[node]:
        node_info_snl[node].append(sno.stem(w))

print("Starting stemming & lemming")    
Parallel(n_jobs = -2, require='sharedmem')(delayed(stemming_lemming)(node) for node in node_info_filtered)

with open('./ISAE_Comp/out/node_info_snl.json', 'w') as file:
    json.dump(node_info_snl, file)
print("Finished s&l on {0} entries of dictionnary and saved it to file".format(len(node_info_snl.keys())))

'''
    Building the full dictionnary of the corpus
'''

dict_full = []
def build_voca(node):
    '''
        Add new word of node to the dictionnary
    '''
    for w in node_info_snl[node]:
        if w not in dict_full:
            dict_full.append(w)

print("Starting building dict")         
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_voca)(node) for node in node_info_snl)

with open('./ISAE_Comp/out/dict.json', 'w') as file:
    json.dump(dict_full, file)
print("Finished building the dictionnary, {0} word entries saved to file".format(len(dict_full)))


'''
    Building bag of word representation fot all the texts
'''

bow = {}
def build_bag(node):
    '''
        Build the BoW representation for node
    '''
    bow[node] = np.zeros(shape=len(dict_full), dtype = int)
    for w in node_info_snl[node]:
        bow[int(node)][dict_full.index(w)] += 1            
    bow[node] = bow[node].tolist()

print("Starting building BoW")
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_bag)(node) for node in node_info_snl)

with open('./ISAE_Comp/out/BOW.json', 'w') as file:
    json.dump(bow, file)
print("Finished building BoW representation, {0} entries saved to file".format(len(bow.keys())))


'''
    Perform PCA to reduce BoW dimension
'''
#import necessary tools
from sklearn.decomposition import PCA
import pickle


#Keep PCA for .95 of the variance
pca = PCA(.95)
#fit PCA on all the data
print("Starting fitting PCA")
pca.fit(list(bow.values()))

pickle.dump(pca, open('./ISAE_Comp/out/pca_model.sav', 'wb'))
print("Finished computing PCA, model saved to file")

#applying PCA reduction to all the BoW
print("Starting PCA transformation")
bow_pca = pca.transform(list(bow.values()))

#back into dict shape
bow_pca = dict(zip(bow.keys(), bow_pca.tolist()))
with open('./ISAE_Comp/out/BOW_pca.json', 'w') as file:
    json.dump(bow_pca, file)
print("Finished applying PCA, {0} entries saved to file".format(len(bow_pca.keys())))

print('Program finished as expected')