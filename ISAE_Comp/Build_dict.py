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
#pickle.dump(dct, open("./ISAE_Comp/out/mydict_full.dict", "wb"))

#filter words that are at least in 2 doc but in less than half of the docs
dct.filter_extremes(no_below=3, no_above=0.20, keep_n = None)
print("Reduced dict contains {0} word".format(len(dct)), flush=True)
#dct.save("./ISAE_Comp/out/reduced_dict.dict")
pickle.dump(dct, open("./ISAE_Comp/out/reduced_dict.dict", "wb"))


