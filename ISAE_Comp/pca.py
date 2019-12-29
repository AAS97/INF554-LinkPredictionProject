'''
    Perform PCA to reduce BoW dimension
'''
#import necessary tools
from sklearn.decomposition import PCA
import pickle
import json

with open('./ISAE_Comp/out/BOW.json', 'r') as file:
    bow = json.load(file)

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
with open('./ISAE_Comp/out/BOW_pca.json', 'w') as file:
    json.dump(bow_pca, file)
print("Finished applying PCA, {0} entries saved to file".format(len(bow_pca.keys())), flush=True)