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

datasets = ['heart1', 'heart2', 'heart3']   # robocze

nDat = len(datasets)
nFolds = 5

skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1410)
scores = np.zeros((nDat, len(clfs), nFolds))

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


tab = {"klasyfikator": ['Bagging', 'Boosting', 'RSE', 'ABoost'],
        "heart1": [meanS(scores[0][0]), meanS(scores[0][1]), meanS(scores[0][2]), meanS(scores[0][3])],
        "heart2": [meanS(scores[1][0]), meanS(scores[1][1]), meanS(scores[1][2]), meanS(scores[1][3])],
        "heart3": [meanS(scores[2][0]), meanS(scores[2][1]), meanS(scores[2][2]), meanS(scores[2][3])],}
print(tabulate(tab, headers="keys"))

np.save(appFolder + '\Scores', scores)