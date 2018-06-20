# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:52:05 2018

@author: ronny
"""
#basics

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import seaborn as sb
from IPython.display import Image
import itertools
from sklearn.metrics import confusion_matrix 
from sklearn import metrics 
import sklearn.cross_validation as cv
import sklearn.tree as tr
import sklearn.svm as svm
import sklearn.ensemble as en
import sklearn.linear_model as lm
import sklearn.metrics as mt
import sklearn.preprocessing as pp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("G:\deep learning\Telkom\Telkom.csv")

data.info()
data.head(20)

#No missing data points 
#replace observation 'No internet' and No phone service with No in the following columns
def replace(frame):
    frame['MultipleLines']=frame['MultipleLines'].replace({'No phone service':'No'})
    frame['OnlineSecurity']=frame['OnlineSecurity'].replace({'No internet service':'No'})
    frame['OnlineBackup']=frame['OnlineBackup'].replace({'No internet service':'No'})
    frame['DeviceProtection']=frame['DeviceProtection'].replace({'No internet service':'No'})
    frame['TechSupport']=frame['TechSupport'].replace({'No internet service':'No'})
    frame['StreamingTV']=frame['StreamingTV'].replace({'No internet service':'No'})
    frame['StreamingMovies']=frame['StreamingMovies'].replace({'No internet service':'No'})
    return frame
df=replace(data)


#Transform the data
def transform(df):
    df['gender']=pd.Categorical(df['gender'])
    df['gender']=df['gender'].cat.codes

    df['Partner']=pd.Categorical(df['Partner'])
    df['Partner']=df['Partner'].cat.codes

    df['Dependents']=pd.Categorical(df['Dependents'])
    df['Dependents']=df['Dependents'].cat.codes
    
    df['PhoneService']=pd.Categorical(df['PhoneService'])
    df['PhoneService']=df['PhoneService'].cat.codes

    df['MultipleLines']=pd.Categorical(df['MultipleLines'])
    df['MultipleLines']=df['MultipleLines'].cat.codes

    df['DeviceProtection']=pd.Categorical(df['DeviceProtection'])
    df['DeviceProtection']=df['DeviceProtection'].cat.codes
    
    df['OnlineSecurity']=pd.Categorical(df['OnlineSecurity'])
    df['OnlineSecurity']=df['OnlineSecurity'].cat.codes
    

    df['OnlineBackup']=pd.Categorical(df['OnlineBackup'])
    df['OnlineBackup']=df['OnlineBackup'].cat.codes
   
    df['TechSupport']=pd.Categorical(df['TechSupport'])
    df['TechSupport']=df['TechSupport'].cat.codes
        
    df['StreamingTV']=pd.Categorical(df['StreamingTV'])
    df['StreamingTV']=df['StreamingTV'].cat.codes
   

    df['StreamingMovies']=pd.Categorical(df['StreamingMovies'])
    df['StreamingMovies']=df['StreamingMovies'].cat.codes
    
    
    df['PaperlessBilling']=pd.Categorical(df['PaperlessBilling'])
    df['PaperlessBilling']=df['PaperlessBilling'].cat.codes
    
    
    df['Contract']=pd.Categorical(df['Contract'])
    df['Contract']=df['Contract'].cat.codes
    

    df['PaymentMethod']=pd.Categorical(df['PaymentMethod'])
    df['PaymentMethod']=df['PaymentMethod'].cat.codes

    df['InternetService']=pd.Categorical(df['InternetService'])
    df['InternetService']=df['InternetService'].cat.codes
    
    df['Churn']=pd.Categorical(df['Churn'])
    df['Churn']=df['Churn'].cat.codes
    return df

df1 =transform(df)

telkom=df1.copy()
telkom
telkom=telkom.drop(['TotalCharges'],axis=1)
telkom=telkom.drop(['Partner'],axis=1)
telkom=telkom.drop(['DeviceProtection'],axis=1)
telkom.info()
telkom=telkom.drop(['C'],axis=1)

telkom.describe()
telkom.info()
def simplify_data(df):
    bins=(-1, 12,24,36,48,60,72)
    group_names=['unknown' 'One year','Two years', 'Three years','Four years','Five years', 'Six years']
    categories=pd.cut(df.tenure, bins, labels=group_names)
    df.tenure=categories.cat.codes
    bins=(-1,15,35,70,89,118)
    gp_names=['Unknown','First_quartile','Second_quartile','Third_quartile','Fourth_quartile']
    categories=pd.cut(df.MonthlyCharges,bins,labels=gp_names)
    df.MonthlyCharges=categories.cat.codes
    return df

new=simplify_data(telkom)
clean=new.drop(['customerID'], axis=1)
new.head(20)
#VISUALIZATION

new.groupby(["MonthlyCharges", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(15,10)) 
new.groupby(["InternetService", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(15,10)) 
new.groupby(["TotalCharges", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(15,10)) 
new.groupby(["gender", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["Contract", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["Partner", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["PaymentMethod", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["PaperlessBilling", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["StreamingMovies", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["OnlineBackup", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["OnlineSecurity", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["DeviceProtection", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["StreamingTV", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["TechSupport", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["MultipleLines", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 
new.groupby(["PhoneService", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(8,10)) 

f, ax=plt.subplots(figsize=(10,8))
sb.heatmap(new.corr(), linewidths=.5, ax=ax)
plt.title('Telkom correlation')
plt.show()
new.corr()
 X=clean.drop(['Churn'],axis=1)
 X.shape
 y=clean['Churn']
 
 from sklearn import neighbors

X.head(10)
X = X.as_matrix().astype(np.float)
X.shape
scaler = pp.StandardScaler()
X = scaler.fit_transform(X)
X


def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    # ii -> train
    # jj -> test indices
    for ii, jj in stratified_k_fold: 
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred
from sklearn import cross_validation
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Logistic Regression:           {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))
print('Random Forest Classifier:      {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('Support vector machine(SVM):   {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))




from sklearn.model_selection import cross_val_predict
clf=svm.SVC(kernel='linear', C=1)
predicted = cross_val_predict(clf, X, y, cv=10)
metrics.accuracy_score(y, predicted) 

from sklearn.model_selection import cross_val_predict
clf=ensemble.RandomForestClassifier
predicted = cross_val_predict(clf, X, y, cv=5)
metrics.accuracy_score(y, predicted) 













gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)

# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
print (gbc.feature_importances_)
feat_importances = pd.Series(gbc.feature_importances_)
feat_importances = feat_importances.nsmallest(18)
feat_importances.plot(kind='barh' , figsize=(10,10)) 

from sklearn.metrics import classification_report, confusion_matrix
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
rf = en.RandomForestClassifier(random_state = 2)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))


n_folds = 5
early_stopping = 10
params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}

xg_train = xgb.DMatrix(x_train, label=y_train);

cv = xgb.cv(params, xg_train, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)


































 y.plot(kind='bar')
 
 y=pd.value_counts(telkom['Churn'], sort = True).sort_index()
 y.plot(kind='bar')
 #standardize data
   
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

pca1 =PCA()
pcafit = pca1.fit_transform(X)
var_explained = pca1.explained_variance_ratio_ #ratio of variance each PC explains
print(pd.Series(var_explained))
print(sum(var_explained[0:13]))

# Process and split data
frame = pd.DataFrame(x_telkom, y_telkom)
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.1, random_state = 50)

x_telkom.astype(int)

## Using statsmodels for its nice output summary
logit= sm.Logit(y, X)
results = logit.fit()
print(results.summary())

plt.rcParams.update(pltdict)
ORs = np.exp(results.params).sort_values();
g = sb.barplot(x = ORs.index, y = ORs-1, palette = 'RdBu_r');
g.set_xticklabels(ORs.index, rotation = 90);
g.set_title('Percent change in odds of leaving\n(i.e., OR minus 1)');
g.set_ylim(-1.10, 1.10);

lr = LogisticRegression(C = 1, random_state=1)
lr.fit(X_train,y_train)
print('10-fold cross validation accuracy: {0:.2f}% \n\n'
      .format(np.mean(cross_val_score(lr, X_train, y_train,cv = 10))*100))
print('Precision/Recall Table: \n')
print(classification_report(y_telkom, lr.predict(x_telkom)))
prc = precision_recall_curve(y_train, lr.decision_function(X_train), pos_label=1);
plt.plot(prc[1],prc[0]);
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

#VISUALIZATION
# Set colors for the different groups
current_palette = sb.color_palette()[0:2]
cols = [current_palette[grp] for grp in telkom.Churn]

sb.set(rc = snsdict)
#plt.rcParams.update(pltdict)
clear()
fig, ax = plt.subplots(3,1, figsize = (5,8));
sb.barplot(data = telkom, x = 'tenure', y= 'Churn', ax = ax[0], color = sb.xkcd_rgb["pale red"]);
sb.regplot(data = telkom, x = 'tenure', y= 'Churn', y_jitter=.02,
            scatter_kws={'alpha':0.05, 'color':cols}, fit_reg = False, ax = ax[1]);
sb.barplot(data = telkom, x= 'tenure', y = 'Churn', ax = ax[2]);
ax[0].set_ylabel("Proportion Leaving"); ax[0].set_xticks([])
ax[2].set_xticklabels(['Stayed','churned']); ax[2].set_ylabel('Mean tenure');
ax[1].set_yticklabels(['Stayed','churned'], fontsize = 24); ax[1].set_ylabel('');ax[1].set_yticks([0,1]);
fig.suptitle('Tenure Plots')
plt.tight_layout()