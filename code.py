import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Importing dataset
data = pd.read_csv("C:/Users/Lenovo/Desktop/DMW_Project/diabetesRaw.csv")

#visualisation of the data

data.hist(figsize=(20,30))
plt.show()

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,title="Diabetic dataset Boxplot",figsize=(20,30))
plt.show()

g=sns.PairGrid(data,vars=['Glucose','Insulin','BMI'],hue="Outcome",height=2.4)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
#g.map(plt.scatter)
g.map_lower(sns.kdeplot,cmap="Blues_d")
plt.show()
##DATA PREPROCESSING##
#removing pregneancies to reduce dimension
data.drop('Pregnancies',axis=1,inplace=True)


#removing the outlier of skin thickness
max_skinthickness=data.SkinThickness.max()
data=data[data.SkinThickness!=max_skinthickness]


#replacing zero values
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]
    
    
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:   
    replace_zero(data, col, 'Outcome')


#print(data)
data.to_csv("C:/Users/Lenovo/Desktop/DMW_Project/movedData.csv",index=False)


# spliting data

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
print("y")
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)

#----------------------------------------confusion  matrix-----------------------------------------------

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


Classifier=KNeighborsClassifier()
Classifier.fit(X_train,y_train)
y_pred=Classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Confusion Matrix of KNN is:\n',cm)

Classifier1=DecisionTreeClassifier()
Classifier1.fit(X_train,y_train)
y_pred=Classifier1.predict(X_test)
cm1=confusion_matrix(y_test,y_pred)
print('Confusion Matrix of DT is:\n',cm1)

Classifier2=GaussianNB()
Classifier2.fit(X_train,y_train)
y_pred=Classifier2.predict(X_test)
cm2=confusion_matrix(y_test,y_pred)
print('Confusion Matrix of NB is:\n',cm2)

Classifier3=AdaBoostClassifier()
Classifier3.fit(X_train,y_train)
y_pred=Classifier3.predict(X_test)
cm3=confusion_matrix(y_test,y_pred)
print('Confusion Matrix of ABC is:\n',cm3)


# helper functions
def train_clf(clf, X_train, y_train):   
    return clf.fit(X_train, y_train)
    
def pred_clf(clf, features, target):   
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label = 1)

def train_predict(clf, X_train, y_train, X_test, y_test):   
    train_clf(clf, X_train, y_train)    
    print("F1 score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)))
    print("F1 score for testing set is: {:.4f}\n".format(pred_clf(clf, X_test, y_test)))



#load algorithms    
nb = GaussianNB()
dtc = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier()
abc = AdaBoostClassifier(random_state=0)

algorithms = [nb,dtc,knn,abc]

for clf in algorithms:     
    print("{}:".format(clf))
    train_predict(clf, X_train, y_train, X_test, y_test)


# In[16]:


from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier(n_neighbors=8)
clf_=knn.fit(X_train,y_train)
y_pred=clf_.predict(X_test)
print('Accuracy of KNN is {}'.format(accuracy_score(y_test,y_pred)*100))




dtc=DecisionTreeClassifier()
clf1_=dtc.fit(X_train,y_train)
y_pred=clf1_.predict(X_test)
print('Accuracy of DT is {}'.format(accuracy_score(y_test,y_pred)*100))

nb=GaussianNB()
clf2_=nb.fit(X_train,y_train)
y_pred=clf2_.predict(X_test)
print('Accuracy of NB is {}'.format(accuracy_score(y_test,y_pred)*100))

abc=AdaBoostClassifier()
clf3_=dtc.fit(X_train,y_train)
y_pred=clf3_.predict(X_test)
print('Accuracy of ABC is {}'.format(accuracy_score(y_test,y_pred)*100))


# In[19]:


#cross-validation score
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf_,X,y,cv=5)
print('CV score of knn:{0:.2f}%+/-{1:.2f}%'.format(scores.mean()*100,scores.std()*200))


# In[20]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf1_,X,y,cv=5)
print('CV score of dtc:{0:.2f}%+/-{1:.2f}%'.format(scores.mean()*100,scores.std()*200))

from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf2_,X,y,cv=5)
print('CV score of nb:{0:.2f}%+/-{1:.2f}%'.format(scores.mean()*100,scores.std()*200))

from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf3_,X,y,cv=5)
print('CV score of abc:{0:.2f}%+/-{1:.2f}%'.format(scores.mean()*100,scores.std()*200))


