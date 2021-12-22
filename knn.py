#!/usr/bin/env python #Shebang
# coding: utf-8

# In[1]:

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from ipywidgets import interact, fixed
import ipywidgets as widgets
from sklearn.metrics import confusion_matrix

# print(pd.__version__) # should be greater than 0.24
# Download dataset from https://www.openml.org/data/v1/download/1586217/banana.arff

# In[2]:

data = arff.loadarff('banana_dataset.arff')
df = pd.DataFrame(data[0])

print(df.head(5))
df['Class']=df['Class'].astype(int)
print(df.head(5))

D=df.to_numpy()
X=df.to_numpy()[:,:2]
y=df.to_numpy()[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=64)

# In[3]:

def knn(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,n_neighbors = 5):
    X = train_data
    y = train_label

    clf = neighbors.KNeighborsClassifier(n_neighbors, n_jobs=-1)
    clf.fit(X, y)

    x1 = np.linspace(-3,3,100)
    x2 = np.linspace(-3,3,100)
    xx,yy = np.meshgrid(x1,x2)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(14,7))
    fig=plt.gcf()
    fig.set_facecolor('white')

    plt.subplot(1,2,1)
    plt.scatter(train_data[train_label==1,0], train_data[train_label==1,1], c='b', marker=".", label='first')
    plt.scatter(train_data[train_label==2,0], train_data[train_label==2,1], c='r', marker="x", label='second')
    plt.legend(loc='upper right')
    plt.contour(xx, yy, Z,linestyles='solid',colors=['black'],levels=['0.5'])
    plt.title('Decision Contours with Train Data and K value = '+str(n_neighbors))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    plt.subplot(1,2,2)
    plt.scatter(test_data[test_label==1,0], test_data[test_label==1,1], c='b', marker=".", label='first')
    plt.scatter(test_data[test_label==2,0], test_data[test_label==2,1], c='r', marker="x", label='second')
    plt.legend(loc='upper right')
    plt.contour(xx, yy, Z,linestyles='solid',colors=['black'],levels=['0.5'])
    plt.title('Decision Contours with Test Data and K value = '+str(n_neighbors))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

# In[4]:

plt.figure(figsize=(14,7))
fig=plt.gcf()  
fig.set_facecolor('white')


plt.subplot(1,2,1)
plt.title('Train Data Distribution')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='b', marker=".", label='first')
plt.scatter(X_train[y_train==2,0], X_train[y_train==2,1], c='r', marker="x", label='second')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.title('Test Data Distribution')
plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c='b', marker=".", label='first')
plt.scatter(X_test[y_test==2,0], X_test[y_test==2,1], c='r', marker="x", label='second')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.legend(loc='upper right')


# In[5]:


k_widget=widgets.IntSlider(value=1,
    min=1,
    max=25,
    step=2,
    description='k',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

interact(knn,n_neighbors=k_widget,
    train_data=fixed(X_train),
    train_label=fixed(y_train),
    test_data=fixed(X_test),
    test_label=fixed(y_test),
    continuous_update=False)


# In[7]:

plt.figure(figsize=(7,7))
k=[1, 3, 5, 9, 15, 20,25]
Accuracy_test=np.zeros(len(k))
Accuracy_train=np.zeros(len(k))
for idx,k_value in enumerate(k):
    clf = neighbors.KNeighborsClassifier(k_value, n_jobs=-1)
    clf.fit(X_train, y_train)
    ypred_train=clf.predict(X_train)
    ypred_test = clf.predict(X_test)

    C_M_test=confusion_matrix(y_test, ypred_test)
    Accuracy_test[idx]=((C_M_test[0,1]+C_M_test[1,0])*100)/np.sum(C_M_test)

    C_M_train=confusion_matrix(y_train, ypred_train)
    Accuracy_train[idx]=((C_M_train[0,1]+C_M_train[1,0])*100)/np.sum(C_M_train)

plt.figure(figsize=(7,7))
fig=plt.gcf()
fig.set_facecolor('white')

plt.plot(k,Accuracy_test,'.r-',label="test data")
plt.plot(k,Accuracy_train,'xb-',label="train data")
plt.ylabel('Error Rate')
plt.xlabel('k')
plt.title("Banana Data: Error Rate vs k")
plt.legend(loc='upper right')
plt.show()