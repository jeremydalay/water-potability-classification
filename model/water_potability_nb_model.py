import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

    if(ans == 1):
        p = 'Potable'
    else:
        p = 'Not Potable'

    new_row = {'ph':par[0],'Hardness':par[1],'Solids':par[2],'Chloramines':par[3],'Sulfate':par[4],'Conductivity':par[5],'Organic_carbon':par[6],'Trihalomethanes':par[7],'Turbidity':par[8],'Prediction':p}
    
    df = pd.read_csv('trial_logs.csv')
    df = df.append(new_row, ignore_index=True)
    df.to_csv('trial_logs.csv')

    return p

def show_cofusion():
    cm = confusion_matrix(y_valid, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors)

def show_corr():
    plt.figure(figsize=(12,8))
    corr_matrix = dataset.corr()
    sns.heatmap(corr_matrix, annot=True, cmap=colors)

def show_pot_count():
    plt.figure(figsize=(12,8))
    plt.title('Potability Count')
    sns.set_style('dark')
    sns.countplot(dataset['Potability'], palette=colors[5:7]) 


colors= sns.color_palette("mako", 10)

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
