from gensim.corpora import Dictionary
import json
import pickle
import time
import numpy as np

from joblib import Parallel, delayed
import multiprocessing
import os

with open('./ISAE_Comp/out/node_info_snl.json','r') as f:
    node_info_snl = json.load(f)
node_info_snl = {int(k):v for k,v in node_info_snl.items()}

texts = node_info_snl.values()
dct = Dictionary(texts)  # initialize a Dictionary
print("Raw dict contains {0} word".format(len(dct)), flush=True)
#dct.save('./ISAE_Comp/out/mydict_full.dict')
pickle.dump(dct, open("./ISAE_Comp/out/mydict_full.dict", "wb"))

#filter words that are at least in 2 doc but in less than half of the docs
dct.filter_extremes(no_below=2, no_above=0.5, keep_n = None)
print("Reduced dict contains {0} word".format(len(dct)), flush=True)
#dct.save("./ISAE_Comp/out/reduced_dict.dict")
pickle.dump(dct, open("./ISAE_Comp/out/reduced_dict.dict", "wb"))

bow = {}
def build_bag(node):
    '''
        Build the BoW representation for node
    '''
    bow[node] = np.zeros(shape=len(dct), dtype = int)
    for ind, freq in dct.doc2bow(node_info_snl[node]):
        bow[int(node)][ind] = freq            
    bow[node] = bow[node].tolist()

print("Starting building BoW", flush=True)
Parallel(n_jobs = -2, require='sharedmem')(delayed(build_bag)(node) for node in node_info_snl)

with open('./ISAE_Comp/out/BOW.json', 'w') as file:
    json.dump(bow, file)
print("Finished building BoW representation, {0} entries saved to file".format(len(bow.keys())), flush=True)




