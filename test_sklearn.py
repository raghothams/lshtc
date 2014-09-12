
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import pylab as pl

# matplotlib.use('Qt4Agg')
iris = datasets.load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=0)
classifier = svm.SVC(kernel='rbf')
y_pred = classifier.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print (cm)
print 'done'

pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

