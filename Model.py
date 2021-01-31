# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:48:59 2021

@author: Mathews
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.feature_selection import SelectPercentile,SelectKBest,SelectFpr
from sklearn.feature_selection import f_classif,mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

original_data=pd.read_csv('loan_data_set.csv')

data=original_data.copy()


##finding Null Values 



###Case 1: Simple Imputer with replacement (mode is used here because of Categorical variables )
imp=SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
temp=imp.fit_transform(data)
data_replaced=pd.DataFrame(data=temp,columns=original_data.columns)

###Case 2: Removing NA Data
data_removed=original_data.copy()
data_removed=data_removed.dropna(axis=0)
print((data_removed.shape[0]/original_data.shape[0])*100,"% of original data is present after Removal")

###Data Imbalance Checking
def DataImbalance(ip_data):
    for column,i in zip(ip_data.columns,range(len(data.columns))):
        # t=ip_data[column].value_counts()
        plt.figure(i)
        sns.countplot(ip_data[column])
    
# Checking for outliers
def OutlierCheck(ip_data):
    plt.figure(figsize=(20,20))
    sns.boxplot(data=pd.DataFrame(ip_data))
    plt.plot()

def removeOutlier(ip_data):
    Q1 = ip_data.quantile(0.25)
    Q3 = ip_data.quantile(0.75)
    IQR = Q3 - Q1
    print(Q1)
    print(ip_data.shape,"Before")
    ip_data = ip_data[~((ip_data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(ip_data.shape,"After")
##Mapping Variable to Numeric Value
def variableMapper(ip_data):
    # columns=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
    ip_data['Gender']=ip_data['Gender'].map({'Male':1,'Female':0})
    ip_data['Property_Area']=ip_data['Property_Area'].map({'Urban':1,'Rural':2,'Semiurban':3})
    ip_data['Married']=ip_data['Married'].map({'Yes':1,'No':0})
    ip_data['Self_Employed']=ip_data['Self_Employed'].map({'Yes':1,'No':0})
    ip_data['Loan_Status']=ip_data['Loan_Status'].map({'Y':1,'N':0})
    ip_data['Education']=ip_data['Education'].map({'Graduate':1,'Not Graduate':0})
    ip_data['Dependents']=ip_data['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
    # for col in columns:
    #     ip_data[col]=ip_data[col].astype("category").cat.codes
        
##b

# I think KNN Model  will be a perfect Model for this Data


def dataExtraction(ip_data):
    #Assigning X variable and Target Varibale
    variableMapper(ip_data)
    X_temp=ip_data.iloc[:,1:12].values
    Y_temp=ip_data.iloc[:,12:14].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_temp,Y_temp, test_size = 0.25,random_state=1, stratify= Y_temp)
    return X_tr, X_te, y_tr, y_te

X_train, X_test, y_train, y_test = dataExtraction(data_replaced)

   
    

###Normalising the Data points
    
X=data_replaced.iloc[:,1:12].values
y=data_replaced.iloc[:,12:14].values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def LogModel(x_tr,x_ts,y_tr):
    logModel=LogisticRegression(max_iter=500)
    logModel.fit(x_tr,y_tr.ravel())
    acc=logModel.score(x_tr,y_tr.ravel())
    y_pre=logModel.predict(x_ts)
    return y_pre,acc
    
#####Logistic Regression  
print("---------------Model 1 (Logistic Regression)-----------------")

y_pred,log_acc=LogModel(X_train,X_test,y_train)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print (classification_report(y_test, y_pred))
print( "Accuracy (Logistic Regression)=",log_acc)

####KNN Model
print("---------------Model 2 (KNN Model)-----------------")

knnModel= KNeighborsClassifier(n_neighbors=18)
knnModel.fit(X_train,y_train.ravel())
knnModel_acc=knnModel.score(X_train,y_train.ravel())
y_pred=knnModel.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print (classification_report(y_test, y_pred))
print( "Accuracy (KNN Model)=",knnModel_acc)


####SVC Model
print("---------------Model 3(SVC  Model)-----------------")

svcModel= SVC()
svcModel.fit(X_train,y_train.ravel())
svcModel_acc=svcModel.score(X_train,y_train.ravel())
y_pred=svcModel.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print (classification_report(y_test, y_pred))
print( "Accuracy (SVC  Model Model)=",svcModel_acc)



# # Create hyperparameter options
# hyperparameters = dict(C=np.logspace(0, 10, 10), penalty=['l1', 'l2'],solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
# print("---------------- model optimization(KNN )---------------")
# best_para_LR = GridSearchCV(LogisticRegression(), hyperparameters, cv=5,n_jobs=-1)
# best_para_LR.fit(X_train,y_train.ravel())
# print("\n Best parameters set found on development set for KNN:")
# print(best_para_LR.best_params_ , "with a score of ",best_para_LR.best_score_)

# print("---------------- End of model optimization(KNN Model )---------------")






# ####Feature Selection Method 1
print("---------------------------Feature Selection--------------------")
X_selected1=SelectPercentile(f_classif, percentile=60).fit_transform(X, y.ravel())
X_train, X_test, y_train, y_test= train_test_split(X_selected1,y, test_size = 0.25,random_state=1, stratify= y)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_pred,log_acc=LogModel(X_train,X_test,y_train)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print (classification_report(y_test, y_pred))
print( "Accuracy (Logistic Regression) Feature 1=",log_acc)
print("----------------------------------------------")

#####Feature Selection Method 2
# X_selected2 = SelectKBest(f_classif, k=6).fit_transform(X, y.ravel())
# X_train, X_test, y_train, y_test= train_test_split(X_selected2,y, test_size = 0.25,random_state=1, stratify= y)
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# y_pred,log_acc=LogModel(X_train,X_test,y_train)
# conf_mat = confusion_matrix(y_test, y_pred)
# print(conf_mat)
# print (classification_report(y_test, y_pred))
# print( "Accuracy (Logistic Regression) Feature 2=",log_acc)

# print("----------------------------------------------")
# #####Feature Selection Method 3- Taking only satistically Important Varibales at alpha=0.05
# X_selected3 = SelectFpr(f_classif, alpha=0.05).fit_transform(X, y.ravel())
# X_train, X_test, y_train, y_test= train_test_split(X_selected3,y, test_size = 0.25,random_state=1, stratify= y)
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
# y_pred,log_acc=LogModel(X_train,X_test,y_train)
# conf_mat = confusion_matrix(y_test, y_pred)
# print(conf_mat)
# print (classification_report(y_test, y_pred))
# print( "Accuracy (Logistic Regression) Feature 3=",log_acc)

# print("----------------------------------------------")


print("----------Ensemble methods-----------")


# cross-fold  model 3
model_cross2 = BaggingClassifier(base_estimator=SVC(),n_estimators=250, random_state=0)
model_cross2.fit(X_train, y_train.ravel())
print("Bagging Classifier",model_cross2.score(X_train,y_train.ravel()))
y_pred=model_cross2.predict(X_test)
print (classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model_scores2=model_selection.cross_val_score(model_cross2,X_train,y_train.ravel(),cv=15)
print(" Bagging Classifiers Cross Val:",model_scores2.mean())
print("-----------------------------------------------------")

# cross-fold  model 4
model_cross3 = DecisionTreeClassifier()
model_cross3.fit(X_train, y_train.ravel())
y_pred=model_cross3.predict(X_test)
print (classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(" Decision tree ",model_cross3.score(X_train,y_train.ravel()))

print("-----------------------------------------------------")

forest = RandomForestClassifier(n_estimators=250, random_state=0)
forest.fit(X_train, y_train.ravel())
y_pred=forest.predict(X_test)
print (classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Random Forrest Classifier",forest.score(X_train,y_train.ravel()))
y_pred=forest.predict(X_test)
print("-----------------------------------------------------")
