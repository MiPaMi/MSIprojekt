from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from adaboost import Adaboost
from sklearn.metrics import accuracy_score
from strlearn.metrics import balanced_accuracy_score
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

def cv52cft(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 5)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1-stats.f.cdf(f, 10, 5)
    return f, p

appFolder = os.path.dirname(os.path.abspath(__file__))

clfs = {
    'ADABoost': Adaboost(n_clf=20),
    'Bagging': BaggingClassifier(base_estimator=SVC(), max_samples=0.7, n_estimators=10, random_state=0),
    'Boosting': GradientBoostingClassifier(),
    'RSE': BaggingClassifier(max_features=0.7, bootstrap=False, bootstrap_features=True)
}

datasets = ['australian', 'banknote', 'breastcan',
            'cryotherapy', 'diabetes', 'ecoli4', 'german', 
            'heart', 'liver', 'monkone'
            ]

nDat = len(datasets)
nFolds = 2
nRep = 5

rskf = RepeatedStratifiedKFold(n_splits=nFolds, n_repeats=nRep, random_state=1410)
scores = np.zeros((nDat, len(clfs), nFolds*nRep))
scores2 = np.zeros((nDat, len(clfs), 2))
scores3 = []

for dataId, dat in enumerate(datasets):
    dat= np.genfromtxt(appFolder + "\datasets\%s.csv" % (dat), delimiter=",")
    X = dat[:, :-1]
    y = dat[:, -1].astype(int)

    for foldId, (train, test) in enumerate(rskf.split(X, y)):
        for clfId, clfName in enumerate(clfs):
            clf = clone(clfs[clfName])
            clf.fit(X[train], y[train])
            yPre = clf.predict(X[test])
            scores[dataId, clfId, foldId] = balanced_accuracy_score(y[test], yPre)
    
for i in range(nDat):
    temp = []
    for j in range(4):
        scores2[i][j] = (round(np.mean(scores[i][j]), 3), round(np.std(scores[i][j]), 3))
        temp.append(str((scores2[i][j][0], scores2[i][j][1])))
    scores3.append(temp.copy())
    temp.clear()


np.save(appFolder + '\Scores', scores)


headers = [ 'ADABoost', 'Bagging', 'Boosting', 'RSE']
names_column = ([['australian'], ['banknote'], ['breastcan'],
                ['cryotherapy'], ['diabetes'], ['ecoli4'], ['german'], 
                ['heart'], ['liver'], ['monkone']
            ])
table = np.concatenate((names_column, scores3), axis=1)
table = tabulate(table, headers)

print("\nPorównanie:\n", table)

alfa = 0.05
fStatistic = np.zeros((len(clfs), len(clfs)))
pValue = np.zeros((len(clfs), len(clfs)))

for k in range(len(datasets)):
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            fStatistic[i, j], pValue[i, j] = cv52cft(scores[k][i], scores[k][j])


    headers = ['ADABoost', 'Bagging', 'Boosting', 'RSE']
    namesColumn = np.array([["ADABoost"], ["Bagging"], ["Boosting"], ["RSE"]])
    fStatisticTable = np.concatenate((namesColumn, fStatistic), axis=1)
    fStatisticTable = tabulate(fStatisticTable, headers, floatfmt=".2f")
    pValueTable = np.concatenate((namesColumn, pValue), axis=1)
    pValueTable = tabulate(pValueTable, headers, floatfmt=".2f")
    print("f-statystyka:\n", fStatisticTable, "\n\np-wartość:\n", pValueTable)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[fStatistic > 0] = 1
    advantageTable = tabulate(np.concatenate(
        (namesColumn, advantage), axis=1), headers)
    print("Przewaga:\n", advantageTable)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[pValue <= alfa] = 1
    significanceTable = tabulate(np.concatenate(
        (namesColumn, significance), axis=1), headers)
    print("Istotnosc:\n", significanceTable)

    statBetter = significance * advantage
    statBetterTable = tabulate(np.concatenate(
        (namesColumn, statBetter), axis=1), headers)
    print("Statystyczna jakość:\n", statBetterTable)