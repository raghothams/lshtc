
from sklearn.datasets import load_svmlight_file
from scipy.io import mmwrite
from numpy import save
import sys

train_csv_file = sys.argv[1]
test_csv_file = sys.argv[2]

X_train, y_train = load_svmlight_file(train_csv_file, multilabel = True)
print type(X_train)
print type(y_train)

mmwrite("X_train.mtx", X_train)
save("y_train.npy", y_train)

del X_train
del y_train

X_test, y_test = load_svmlight_file(test_csv_file, multilabel = True)
print type(X_test)
print type(y_test)

mmwrite("X_test.mtx", X_test)
save("y_test.npy", y_test)

