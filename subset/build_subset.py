import json
import numpy as np
import pickle

with open('./ISAE_Comp/out/node_info_snl.json','r') as f:
    node_info_snl = json.load(f)
node_info_snl = {int(k):v for k,v in node_info_snl.items()}

subset_index = list(np.random.randint(0, len(node_info_snl), 5000))


#making sure that subset has at least all testing nodes
with open('./data/testing.txt','r') as file:
    for line in file:
        line = line.split()
        if int(line[0]) not in subset_index:
            subset_index.append(int(line[0]))
        if int(line[1]) not in subset_index:
            subset_index.append(int(line[1]))

with open('./subset.json','wb') as file:
    pickle.dump(subset_index, file)