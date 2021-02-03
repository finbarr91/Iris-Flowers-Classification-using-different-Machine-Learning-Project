# Iris-Flowers-Classification-using-different-Machine-Learning-Project
This is a Supervised machine learning Algorithms (Logistic Regression,Linear Discriminant Analysis, KNeighbors Classifier, Decision Tree Classifier, GaussianNB,SVM)written in Python 
import pandas as pd
import numpy as np
import sklearn

from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('iris.data')
print(df.head())
print(df.describe())
print(df.shape)
# box and whisker plots
df.plot(kind = 'box', subplots=True, layout = (2,2))
plt.show()


df.hist()
plt.show()

scatter_matrix(df)
plt.show()

x = np.array(df[df.columns[0:4]])
y = np.array(df[df.columns[4]])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=1, shuffle=True)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
# Evaluate predictions
print('accuracy:' ,accuracy_score(y_test, predictions))
print('confusion matrix :' , confusion_matrix(y_test, predictions))
print('classification_report :' , classification_report(y_test, predictions))
