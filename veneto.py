##author: Andrea Ceolin
##date: December 2017


import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import Counter

#Load the data
data = [line.split() for line in open('Veneto_data.csv')]
#Remove missing entries '-'
data = [[word for word in dialect if word != '-'] for dialect in data]
#Create a vector with all the words of the database
wordlist = list(set([word for dialect in data for word in dialect[3:]]))

#Labels. We have five areas represented: Central, Coastal, North, Venice, West
y = [dialect[0] for dialect in data]
print("Labels of the regional varieties we are trying to predict:")
print(set(y))

#Languages
langs = [(dialect[1], dialect[2]) for dialect in data]
print("Location of the datapoint:")
print(set(langs))

#Feature matrix. 1 means that a word is recorded for a language, 0 meand that a word is not recorded
X = [[1 if word in dialect[3:] else 0 for word in wordlist] for dialect in data]

model = LogisticRegression()
model.fit(X,y)
ypred = model.predict(X)

accuracy = []
error_dic = Counter()
#Accuracy calculated over 100 trials
for i in range(100):
    #Here we want to shuffle, keeping the indeces, 1) the instances, 2) their labels,
    # 3) the name of the languages, so that we can retrieve the most common mistakes

    p = [x for x in range(len(y))]
    q = [x for x in range(len(X[0]))]

    random.shuffle(p)
    random.shuffle(q)

    #shuffle everything
    X_features = [[X[i][j] for j in q] for i in p]
    y_shuffled = [y[i] for i in p]
    langs_shuffled = [langs[i] for i in p]

    nTrain = int(0.7*len(y))
    xTrain = int(0.1*len(X[0]))

    #Split the data
    Xtrain = np.matrix([x_matrix[:xTrain] for x_matrix in X_features[:nTrain]])
    ytrain = np.array(y_shuffled[:nTrain])
    Xtest = np.matrix([x_matrix[:xTrain] for x_matrix in X_features[nTrain:]])
    ytest = np.array(y_shuffled[nTrain:])

    #This allows us to see which languages are misclassified
    langs_test = langs_shuffled[nTrain:]

    modelLogistic = LogisticRegression()
    modelLogistic.fit(Xtrain,ytrain)
    ypred = modelLogistic.predict(Xtest)
    accuracyLogistic = accuracy_score(ytest, ypred)
    accuracy.append(accuracyLogistic)
    errors = []
    for index, label in enumerate(ytest):
        if ytest[index] != ypred[index]:
            errors.append((langs_test[index], ytest[index], ypred[index]))
    if errors:
        for error in errors:
            error_dic[error] += 1

print("#######################")
print("SUMMARY")
print("#######################")
print("Mean accuracy:" + str(np.mean(accuracy)))
print("Errors:")
for error in error_dic.most_common(10):
    print(error)

