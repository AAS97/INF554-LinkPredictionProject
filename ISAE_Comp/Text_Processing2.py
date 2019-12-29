import pandas as pd
import numpy as np
import json
from joblib import Parallel, delayed
import multiprocessing
import os


def subset_link(nb_node):
    with open('./data/training.txt', 'r') as f :
        testing_set = [] 
        for line in f:
            line = line.split()
            testing_set.append(line)
    dataset = pd.DataFrame(testing_set, columns=["Node1","Node2","Edge"])
    dataset = dataset.astype(np.int)

    sub_training = dataset.sample(n = nb_node, random_state=0)
    #node_list.remove_duplicate


subset_link(10)

with open('./ISAE_Comp/out/node_info_snl.json','r') as f:
    node_info_snl = json.load(f)

node_info_snl_sub = {}
def populate_sub_dict(node):
    node_info_snl_sub[node] = node_info_snl[node]

Parallel(n_jobs = -2, require='sharedmem')(delayed(populate_sub_dict)(node) for node in node_list)
'''
    Building the full dictionnary of the corpus
'''

dict_full = []
def build_voca(node):
    '''
        Add new word of node to the dictionnary
    '''
    for w in node_info_snl_sub[node]:
        if w not in dict_full:
            dict_full.append(w)

print("Starting building dict")         
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_voca)(node) for node in node_list)

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
    for w in node_info_snl_sub[node]:
        bow[int(node)][dict_full.index(w)] += 1            
    bow[node] = bow[node].tolist()

print("Starting building BoW")
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_bag)(node) for node in node_list)

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