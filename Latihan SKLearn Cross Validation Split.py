#Pada codelab kali ini kita akan menggunakan cross_validation_score pada classifier decision_tree. Dataset yang digunakan adalah dataset iris.

import sklearn
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import tree
#load iris data set
iris = datasets.load_iris()

#definisi atribut dan label pada datasetd
x=iris.data
y=iris.target

#membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

# mengevaluasi performa model dengan cross_val_score
scores = cross_val_score(clf, x, y, cv=5)
print(scores)