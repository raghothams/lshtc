#!/home/lroot/anaconda/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory, Parallel, delayed
from tempfile import mkdtemp
from sklearn.naive_bayes import MultinomialNB


# function to read train data
# this function call has to be memory mapped by parent 
#   so that children do not create copies
def get_train_data(file_path):
    print "will read now"
    X,y = load_svmlight_file(file_path, multilabel=True)

    return X,y


# dummy function to call memory mapped function to read training data
def child_mode(mem_cached_func, train_file_path, test_file_path, i):
    t0 = time.time()
    X_train, y_train = mem_cached_func(train_file_path)
    print "train load Done in ", time.time() - t0
    print test_file_path+"."+str(i)

    clf = MultinomialNB()
    t0 = time.time()
    clf.fit(X_train, y_train)
    print time.time()-t0

    t0 = time.time()
    X_test, y_test = load_svmlight_file(test_file_path+"."+str(i), 
                        multilabel = True, n_features=X_train.shape[1])
    print "test load Done in ", time.time() - t0

    t0 = time.time()
    Z = clf.predict(X_test)
    print "predicted in ", time.time() - t0

    np.savetxt('res.txt.'+str(i), Z, delimiter=" ", fmt="%s")
    print "saved result"


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "Usage: \nmemmap <train-file-path> <test-file-path>\n"
        exit("No file path specified")

    train_file_path = sys.argv[1]
    print "file to be read ",train_file_path
    
    test_file_path = sys.argv[2]
    print "file to be read ",test_file_path

# get a temp directory to cache the array
    cache_dir = mkdtemp()

# get Memory instance
    mem = Memory(cachedir=cache_dir, mmap_mode='r', verbose=5)
    memmed_getter = mem.cache(get_train_data)

    Parallel(n_jobs=3, verbose=3) (delayed(child_mode)\
            (memmed_getter, train_file_path, test_file_path, i) \
            for i in range(1,4))

    del memmed_getter
    del mem
    del cache_dir

