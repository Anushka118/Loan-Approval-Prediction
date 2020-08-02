#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
print(train.columns)


# In[3]:


#categorical variables-1
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize = True).plot.bar(figsize = (15,15),title = 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize = True).plot.bar(figsize = (15,15),title = 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize = True).plot.bar(figsize = (15,15),title = 'Self_Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize = True).plot.bar(figsize = (15,15),title = 'Credit_History')
plt.show()


# In[4]:


#categorical variables- ordinal variables
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title = 'Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize= True).plot.bar(figsize=(24,6),title = 'Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize= True).plot.bar(figsize=(24,6),title = 'Property_Area')
plt.show()


# In[5]:


#numerical Variables - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
plt.figure(1)
plt.subplot(321)
sns.distplot(train['ApplicantIncome']);
plt.subplot(322)
train['ApplicantIncome'].plot.box(figsize=(24,15))

plt.subplot(323)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(324)
train['CoapplicantIncome'].plot.box(figsize=(24,15))

plt.subplot(325)
sns.distplot(train['LoanAmount']);
plt.subplot(326)
train['LoanAmount'].plot.box(figsize=(24,15))
plt.show()


# In[6]:


Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Married = pd.crosstab(train['Married'],train['Loan_Status'])
Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
Education = pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()


# In[7]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.show()


# In[8]:


Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True,figsize=(4,4))
plt.show()


# In[9]:


bins = [0,2500,4000,6000,8100]
group  = ['Low','Average','High','Very High']
train['Income_bin'] = pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')
plt.show()


# In[10]:


bins = [0,1000,3000,4200]
group = ['Low','Average','High']
train['CoAppIncBin'] = pd.cut(train['CoapplicantIncome'],bins,labels=group)
CoAppIncBin = pd.crosstab(train['CoAppIncBin'],train['Loan_Status'])
CoAppIncBin.div(CoAppIncBin.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True)
plt.xlabel('CoAppIncBin')
P = plt.ylabel('Percentage')
plt.show()


# In[11]:


train['TotalInc'] = train['ApplicantIncome'] + train['CoapplicantIncome']
bins = [0,2500,4000,6000,8100]
group = ['Low','Average','High','Very High']
train['TotalIncBin'] = pd.cut(train['TotalInc'],bins,labels=group)
TotalIncBin = pd.crosstab(train['TotalIncBin'],train['Loan_Status'])
TotalIncBin.div(TotalIncBin.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True)
plt.xlabel('TotalIncBin')
P = plt.ylabel('Percentage')
plt.show()


# In[12]:


bins = [0,100,200,700]
group=['Low','Average','High']
train['LoanAmtBin'] = pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmtBin = pd.crosstab(train['LoanAmtBin'],train['Loan_Status'])
LoanAmtBin.div(LoanAmtBin.sum(1).astype(float),axis=0).plot(kind="bar",stacked = True)
plt.xlabel('LoanAmtBin')
P = plt.ylabel('Percentage')
plt.show()


# In[13]:


train = train.drop(['Income_bin','CoAppIncBin','TotalIncBin','LoanAmtBin'],axis = 1)


# In[14]:


train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)
train['Loan_Status'].replace('Y',1,inplace=True)

matrix = train.corr()
f, ax = plt.subplots(figsize=(4,3))
sns.heatmap(matrix,vmax=0.8,square = True,cmap="BuPu");
plt.show()


# In[15]:


print("Train Dataset:")
print(train.isnull().sum())
print("Test Dataset:")
print(test.isnull().sum())


# In[16]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)
train['Married'].fillna(train['Married'].mode()[0],inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)


# In[17]:


test['Gender'].fillna(train['Gender'].mode()[0],inplace = True)
test['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)


# In[18]:


train['LoanAmtLog'] = np.log(train['LoanAmount'])
test['LoanAmtLog'] = np.log(test['LoanAmount'])


# In[19]:


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

x = train.drop('Loan_Status',1)
y = train.Loan_Status

x = pd.get_dummies(x)
train = pd.get_dummies(train);
test=pd.get_dummies(test)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.3)
scaler = preprocessing.StandardScaler().fit(x_train)
scaler.transform(x_train)


# In[20]:


#decision tree
from sklearn import tree
decTree = tree.DecisionTreeClassifier()
decTree.fit(x_train,y_train)
y_pred_decision_Tree = decTree.predict(x_test)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
kNbr = KNeighborsClassifier(n_neighbors = 3)
kNbr.fit(x_train,y_train)
y_pred_kNbr = kNbr.predict(x_test)


# In[22]:


#NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred_gnb = gnb.predict(x_test)


# In[23]:


#logistic regression
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred_logistic_regression = logReg.predict(x_test)


# In[24]:


from sklearn.metrics import accuracy_score
acc_decision_tree = accuracy_score(y_test,y_pred_decision_Tree)
acc_kNbr = accuracy_score(y_test, y_pred_kNbr)
acc_gnb = accuracy_score(y_test, y_pred_gnb)
acc_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
print("accuracy of decision tree model:",acc_decision_tree)
print("accuracy of KNeighbors:",acc_kNbr)
print("accuracy of Gaussian Naive Bayes:",acc_gnb)
print("accuracy of Logistic Regression:",acc_logistic_regression)


# In[25]:


test['TotalInc'] = test['ApplicantIncome'] + test['CoapplicantIncome']


# In[27]:


y_pred_decision_Tree_test = decTree.predict(test)
y_pred_kNbr_test = kNbr.predict(test)
y_pred_gnb_test = gnb.predict(test)
y_pred_logistic_regression_test = logReg.predict(test)


# In[33]:


print("test data prediction using decision tree:")
count = 1
for i in y_pred_decision_Tree_test:
    if i==1:
        print(count,"yes")
    else:
        print(count,"no")
    count+=1


# In[39]:


print("test data prediction using k-neighbors:")
count = 1
for i in y_pred_kNbr_test:
    if i==1:
        print(count,"yes")
    else:
        print(count,"no")
    count+=1


# In[37]:


print("test data prediction using Gaussian Naive Bayes:")
count = 1
for i in y_pred_gnb_test:
    if i==1:
        print(count,"yes")
    else:
        print(count,"no")
    count+=1


# In[38]:


print("test data prediction using Logistic Regression:")
count = 1
for i in y_pred_logistic_regression_test:
    if i==1:
        print(count,"yes")
    else:
        print(count,"no")
    count+=1


# In[ ]:




