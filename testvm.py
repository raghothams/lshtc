
from sklearn.svm import LinearSVC
from scipy.io import mmread
from numpy import load
import time

t0 = time.time()
X_train, y_train = load_svmlight_file("../../wiki_large2.0/small-train.csv", 
                                multilabel = True)
print "train load Done in ", time.time() - t0

t0 = time.time()
X_test, y_test = load_svmlight_file("../../wiki_large2.0/small-test.csv", 
                                multilabel = True, n_features=X_train.shape[1])
print "test load Done in ", time.time() - t0

clf = LinearSVC()

t0 = time.time()
clf.fit(X_train, y_train)
print time.time()-t0

t0 = time.time()
Z = clf.predict(X_test)
print time.time()-t0
print type(Z)

