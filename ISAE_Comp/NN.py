from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import csv

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Dense
X = []
y = []
with open("training_NN.csv","r") as f:
    reader = csv.reader(f)
    for line in reader:
        X.append(np.array(line[0:5]))
        y.append(np.array(line[6]))

#load data metrics
print(X[1])
X = np.array(X[1:]).astype(float)
#load data classification
y = np.array(y[1:]).astype(float)

#normalise the data
data_scaled = preprocessing.scale(X)

# define the nn model
model = Sequential()
model.add(Dense(activation='relu', input_shape=(5,)))
model.add(Dense(activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model = KerasClassifier(model)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.33, random_state=42)
print("Training set legnth : {0}   Testing set length : {1}".format(len(X_train), len(X_test)), flush=True)

#model.fit(X_train, y_train, epochs=10, batch_size=32)

#model.save("model.h5")
#print("Saved model to disk")

#core = model.evaluate(X_test, y_test, batch_size=32)
#print(score)

epochs = np.array([10, 20, 50])
batches = np.array([5, 10, 20, 30])
param_grid = dict(nb_epoch=epochs, batch_size=batches)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))