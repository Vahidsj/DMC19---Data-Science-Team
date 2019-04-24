import os
import glob

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

path = "C:/Users/Vahid/Desktop/DMC_2019_task/"

for path_fileName in glob.glob(path + '*.csv'):
    if 'train' in path_fileName:
        Table_Train = pd.read_csv(path_fileName, sep='|')
    elif 'test' in path_fileName:
        Table_Test = pd.read_csv(path_fileName, sep='|')
    else:
        print('No valid CSV-Files in this folder: {}'.format(path))

print('Training Set:')
print(Table_Train.dtypes)
print('Number of rows:', Table_Train.shape[0])
print('Number of columns:', Table_Train.shape[1])
Table_Train.head(3)


print('Test Set:')
print(Table_Test.dtypes)
print('Number of rows:', Table_Test.shape[0])
print('Number of columns:', Table_Test.shape[1])
Table_Test.head(3)


table_train = Table_Train.drop(columns=['fraud'])
table_train.describe()

Table_Test.describe()

Table_Train.isnull().values
print(Table_Train.isnull().values.any())
print(Table_Test.isnull().values.any())

X = Table_Train.values[:, 0:9]
y = Table_Train.values[:, 9]
n_features = X.shape[1]
n_samples = X.shape[0]
n_classes = len(np.unique(y))

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

################################################################################# Data Cleansing







################################################################################ Feature Enginering







################################################################################# Create Training and Test set

random_state = np.random.RandomState(0)
X, y = shuffle(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=random_state)

################################################################################# Classifier










################################################################################# Validation - Confusion_matrix










################################################################################ Johannes












################################################################################ Alisa












################################################################################ Markus











################################################################################ Thomas












################################################################################ Manuel












################################################################################ Vahid












################################################################################ Shiwen












################################################################################ Jannis












################################################################################ Georg












################################################################################

