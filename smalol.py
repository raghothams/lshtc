import logging
from sklearn.datasets import load_svmlight_file
from sklearn import svm
import time

logging.basicConfig(filename='result-small.log',level=logging.DEBUG)

t0 = time.time()
X,y = load_svmlight_file("/home/lroot/code/lshtc/data/med-train.csv", multilabel=True)
logging.info("loaded train data in : %s" % (time.time() - t0))

clf = svm.SVC(kernel='linear')
t0 = time.time()
logging.info("will fit clf now")

clf.fit(X, y)
logging.info('done fitting train data in : %s' % (time.time() - t0))

t0 = time.time()
logging.info("will load test")
X_test, y_test = load_svmlight_file("/home/lroot/code/lshtc/data/small-test.csv", multilabel=True, n_features=X.shape[1])
logging.info('done loading test  data in : %s' % (time.time() - t0))

t0 = time.time()
logging.info("will predict")
Z = clf.predict(X_test)
logging.info('done predicting test  data in : %s' % (time.time() - t0))

np.savetxt('res-small.txt', Z, delimiter=" ", fmt="%s")

