import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def ph(x):
    if x<6.5:
        return 0
    elif x<=8.5 and x>=6.5:
        return 1
    else:
        return 2

def hard(x):
    if x<60:
        return 0
    elif x<=120 and x>=60:
        return 1
    else:
        return 2

def tds(x):
    if x<50:
        return 0
    elif x<=150 and x>=50:
        return 1
    else:
        return 2

def chlr(x):
    if x<=4:
        return 0
    else:
        return 1

def slft(x):
    if x<250:
        return 0
    else:
        return 1

def cdty(x):
    if x<=400:
        return 0
    else:
        return 1

def crbn(x):
    if x<4:
        return 0
    else:
        return 1

def thts(x):
    if x<=80:
        return 0
    else:
        return 1

def tbty(x):
    if x<5:
        return 0
    else:
        return 1

def pred(par):
    cond = [ph,hard,tds, chlr, slft, cdty, crbn, thts, tbty]
    vals = []
    for i in range(len(par)):
        vals += [cond[i](par[i])]
    ans = NB.predict([vals])
    return ans

dataset = pd.read_csv('..\dataset\water_potability_final.csv')

X = dataset.drop(['Potability'], axis=1)
y = dataset['Potability']

cond = [ph,hard,tds, chlr, slft, cdty, crbn, thts, tbty]
col = X.columns.values
data = pd.DataFrame(columns=col)

for i in range(len(col)):
    data[col[i]] = dataset[col[i]].apply(cond[i])
X = data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.30, random_state= 41)
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_valid)
acc = round(accuracy_score(y_valid, y_pred), 4)
