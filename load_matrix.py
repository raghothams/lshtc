
from scipy.io import mmread
from numpy import load

X_test = mmread("X_train.mtx")
print type(X_test)

y_test = load("y_train.npy")
print type(y_test)

X_test = mmread("X_test.mtx")
print type(X_test)

y_test = load("y_test.npy")
print type(y_test)

