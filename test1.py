
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2

import numpy as np
import time

t0 = time.time()

X_train, y_train = load_svmlight_file("data/tiny-train.csv", 
                                multilabel = True)
print "load Done in ", time.time() - t0
clf = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree')
clf.fit(X_train, y_train)

t0 = time.time()
print "fit Done in ", time.time() - t0

t0 = time.time()
X_test, y_test = load_svmlight_file("data/tiny-test.csv", 
                                multilabel = True, n_features=X_train.shape[1])
print "test load Done in ", time.time() - t0
t0 = time.time()
ch2 = SelectKBest(chi2, k=5)
X_train = ch2.fit(X_train, y_train)
print "fit transform Done in ", time.time() - t0
"""
t0 = time.time()
X_test = ch2.transform(X_test)
print "tranform test Done in ", time.time() - t0
"""
# print X_test.shape
#print dir(X_test)
t0= time.time()
Z = clf.predict(X_test)
print time.time() - t0

