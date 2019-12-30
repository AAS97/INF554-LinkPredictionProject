from gensim.corpora import Dictionary
import json
import pickle
import time
import numpy as np

from joblib import Parallel, delayed
import multiprocessing
import os



reduced_dict = pickle.load(open("./ISAE_Comp/out/reduced_dict.dict", "rb"))
with open('./ISAE_Comp/out/node_info_snl.json','r') as f:
    node_info_snl = json.load(f)
node_info_snl = {int(k):v for k,v in node_info_snl.items()}

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

#with open('./ISAE_Comp/out/BOW.json', 'w') as file:
#    json.dump(bow, file)
print("Finished building BoW representation", flush=True)



'''
    Perform PCA to reduce BoW dimension
'''
#import necessary tools
from sklearn.decomposition import PCA
import pickle
import json

#with open('/home/promo20/a.d-andoque/BOW.json', 'r') as file:
#    bow = json.load(file)

#Keep PCA for .95 of the variance
pca = PCA(.95)
#fit PCA on all the data
print("Starting fitting PCA", flush=True)
pca.fit(list(bow.values()))

pickle.dump(pca, open('./ISAE_Comp/out/pca_model.sav', 'wb'))
print("Finished computing PCA, model saved to file", flush=True)

#applying PCA reduction to all the BoW
print("Starting PCA transformation")
bow_pca = pca.transform(list(bow.values()))

#back into dict shape
bow_pca = dict(zip(bow.keys(), bow_pca.tolist()))
with open('/home/promo20/a.d-andoque/BOW_pca.json', 'w') as file:
    json.dump(bow_pca, file)
print("Finished applying PCA, {0} entries saved to file".format(len(bow_pca.keys())), flush=True)
