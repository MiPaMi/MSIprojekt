from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from adaboost import Adaboost
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import os
from sklearn.base import clone
from math import *
from scipy.stats import ttest_ind
from scipy import stats
from sklearn.svm import SVC


def meanS(score):
    meanScore = np.mean(score)
    stdScore = np.std(score)
    r = (round(meanScore, 3), round(stdScore, 2))
    return r

appFolder = os.path.dirname(os.path.abspath(__file__))

clfs = {
    'Bagging': BaggingClassifier(base_estimator=SVC(), max_samples=0.5, n_estimators=10, random_state=0),
    'Boosting': GradientBoostingClassifier(),
    'RSE': BaggingClassifier(max_features=0.5, bootstrap=False, bootstrap_features=True),
    'ADABoost': Adaboost()
}

datasets = ['australian', 'banknote', 'breastcan',
            'cryotherapy', 'diabetes', 'ecoli4', 'german', 
            'heart', 'liver', 'monkone'
            ]

nDat = len(datasets)
nFolds = 5

skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1410)
scores = np.zeros((nDat, len(clfs), nFolds))
scores2 = np.zeros((nDat, len(clfs), 2))

for dataId, dat in enumerate(datasets):
    dat= np.genfromtxt(appFolder + "\datasets\%s.csv" % (dat), delimiter=",")
    X = dat[:, :-1]
    y = dat[:, -1].astype(int)

    for foldId, (train, test) in enumerate(skf.split(X, y)):
        for clfId, clfName in enumerate(clfs):
            clf = clone(clfs[clfName])
            clf.fit(X[train], y[train])
            yPre = clf.predict(X[test])
            scores[dataId, clfId, foldId] = accuracy_score(y[test], yPre)
    
    for i in range(nDat):
        for j in range(4):
            scores2[i][j] = (round(np.mean(scores[i][j]), 3), round(np.std(scores[i][j]), 3))

np.save(appFolder + '\Scores', scores)


headers = ['Bagging', 'Boosting', 'RSE', 'ADABoost']
names_column = ([['australian'], ['banknote'], ['breastcan'],
                ['cryotherapy'], ['diabetes'], ['ecoli4'], ['german'], 
                ['heart'], ['liver'], ['monkone']
            ])
table = np.concatenate((names_column, scores2), axis=1)
table = tabulate(scores2, headers)

print("\nPorównanie:\n", table)