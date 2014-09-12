
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_svmlight_file
import traceback
import time

t0 = time.time()
X_train, y_train = load_svmlight_file("/home/lroot/code/lshtc/data/tiny-train.csv",multilabel=True)
print time.time() - t0

try:
    clf = SGDClassifier()
    t0 = time.time()
    clf.partial_fit(X_train, y_train, classes=[y_train[1],y_train[2]])
    print time.time() - t0
     
    X_test, y_test = load_svmlight_file("/home/lroot/code/lshtc/data/tiny-test.csv",
				    multilabel=True, n_features=10259)
    t0 = time.time()
    z = clf.predict(X_test)

except Exception as e:
    traceback.print_exc()
print time.time() - t0


