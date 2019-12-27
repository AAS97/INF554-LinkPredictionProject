from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import json

with open('./ISAE_Comp/out/BOW_pca.json', 'r') as file:
    bow = json.load(file)
bow = {int(k):v for k,v in bow.items()}

X = []
y = []

with open('./data/training.txt', 'r') as file:
    for line in file:
        line = line.split()
        i = int(line[0])
        j = int(line[1])
        feat = bow[i] + bow[j]
        X.append( feat )
        y.append(line[2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Training set legnth : {0}   Testing set length : {1}".format(len(X_train), len(X_test)))


mlp = MLPClassifier(verbose=True)

mlp.fit(X_train,y_train)

prediction = list(mlp.predict(X_test))

with open('./ISAE_Comp/out/prediction.json',"w") as file:
    json.dump(prediction, file)
print("Prediction wrote to file")

print("Getting {0} % accuracy".format(accuracy_score(y_test, prediction, normalize=False)/len(X_test)))
print("Confusion Matrix")
print(confusion_matrix(y_test, prediction)/len(X_test))



'''
X_test = []
with open('./data/testing.txt', 'r') as file:
    for line in file:
        line = line.split()
        if int(line[0]) < max_node and int(line[1]) < max_node:
            X_test.append( bow[int(line[0])].extend(bow[int(line[1])]) )
print(X_test)           
'''