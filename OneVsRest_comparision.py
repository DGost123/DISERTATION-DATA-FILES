#!/usr/bin/env python
# coding: utf-8

# In[77]:


import matplotlib.pyplot as plt


# In[78]:


name_string = '20210514'
# https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff


# In[79]:


import pandas as pd
import pyodbc

# Environment settings: 
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[80]:


# set up connection to the SQL server
conn = pyodbc.connect('Driver={SQL Server};'
'Server=MCS-TEST-SQL;'
'Database=MCS;'
'Trusted_Connection=yes;')


# In[81]:


# SQL query
defaults_sql = '''
SELECT * FROM Dataset

'''


# In[82]:


# execute query and save results to Pandas dataframe
defaults = pd.read_sql(defaults_sql.format('[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]'), conn)
defaults.shape


# In[83]:


# and export it as a csv
defaults.to_csv('base_table_' + name_string + '.csv', index=False)


# In[84]:


# import stuff for modeling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[85]:


# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[86]:


# get the dataset
def get_dataset():
	X = defaults.drop(columns = ['vie', 'du', 'tr', 'ket', 'pen', 'ses', 'sep', 'ast'])
	X.shape
	y = defaults.drop(columns = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay'
	,'Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])
	y.shape
	return X, y



# In[87]:


# load dataset
X, y = get_dataset()
# class_names = ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential, 'global']
               


# In[88]:


X.describe()


# In[89]:


y.describe()


# In[90]:


# create dataframe from file
dataframe = X

# use corr() method on dataframe to
# make correlation matrix
matrix = dataframe.corr()
 
# print correlation matrix
print("Correlation Matrix is : ")
print(matrix)

corr = dataframe.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps


# In[91]:


from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import label_binarize


# In[92]:


n_inputs, n_outputs = X.shape[1], y.shape[1]
print(n_inputs, n_outputs)


# In[93]:


# Binarize the output
y = label_binarize(y, classes=['vie', 'du', 'tr', 'ket', 'pen', 'ses', 'sep', 'ast'])
n_classes = y.shape[1]


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[95]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pylab as pl
from sklearn.metrics import roc_curve, auc


# In[ ]:





# In[ ]:





# In[348]:


# Learn to predict each class against the other
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif =classif.fit(X_train, y_train)


# In[349]:


yhat = classif.predict(X_test)


# In[350]:


print(yhat)


# In[351]:


#You need to consider the decision_function and the prediction separately. The decision is the distance from the
#hyperplane to your sample. This means by looking at the sign you can tell if your sample is located right or left 
#to the hyperplane. So negative values are perfectly fine and indicate the negative class ("the other side of the 
#hyperplane")
#https://stackoverflow.com/questions/46820154/negative-decision-function-values


# In[352]:


#Predict margin (libsvm name for this is predict_values)
y_score = classif.decision_function(X_test)
print(y_score)


# In[353]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[354]:


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[355]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[356]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[357]:


# Plot ROC curve
#the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one.
#This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
#The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate
#while minimizing the false positive rate.

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[358]:


#ROC curves are typically used in binary classification to study the output of a classifier. 
#In order to extend ROC curve and ROC area to multi-class or multi-label classification, it is necessary
#to binarize the output. One ROC curve can be drawn per label, but one can also draw a ROC curve by considering
#each element of the label indicator matrix as a binary prediction (micro-averaging).


# In[359]:


#In a multilabel classification setting, sklearn.metrics.accuracy_score only computes the subset
#accuracy i.e. the set of labels predicted for a sample must exactly match the corresponding
#set of labels in y_true. This is exact match ratio 


# In[360]:


#Hamming score  (since it is closely related to the Hamming loss), or label-based accuracy


# In[361]:


from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhat
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


# In[ ]:





# In[362]:


#The F1 average assumes that precision and recall are equally weighted. But this is untrue in reality. 
#Use the averaged precision and recall to calculate the F1 score makes more sense since that will better 
#reflect your favor on precision or recall. 
#http://sentiment.christopherpotts.net/classifiers.html#assessment 


# In[363]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

m = MultiLabelBinarizer().fit(y_true)

f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')


# In[364]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')


# In[365]:


precision_recall_fscore_support(y_true, y_pred, average='micro')


# In[366]:


precision_recall_fscore_support(y_true, y_pred, average='weighted')


# In[367]:


#When true positive + false positive == 0, precision is undefined. When true positive + 
#+false negative == 0, recall is undefined. In such cases, by default the metric will be set to 0,
#as will f-score, and UndefinedMetricWarning will be raised. This behavior can be modified with
#zero_division.
#'micro':Calculate metrics globally by counting the total true positives, false negatives and
#false positives.
#'macro':Calculate metrics for each label, and find their unweighted mean. This does not take
#label imbalance into account.
#'weighted':Calculate metrics for each label, and find their average weighted by support
#(the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.


# In[368]:


#it is possible to compute per-label precisions, recalls, F1-scores and supports instead of 
#averaging


# In[369]:


precision_recall_fscore_support(y_true, y_pred)


# In[370]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#this is the most commonly used strategy and is a fair default choice.


# In[ ]:





# In[ ]:





# In[371]:


from sklearn.linear_model import Perceptron
clf= OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0))
clf=clf.fit(X_train, y_train)


# In[372]:


yhatt = clf.predict(X_test)
print(yhatt)


# In[373]:


#Predict margin (libsvm name for this is predict_values)
y_scoree = clf.decision_function(X_test)
#print(y_score)


# In[374]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scoree[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[375]:


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scoree.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[376]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[377]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[378]:


# Plot ROC curve
#the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one.
#This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
#The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate
#while minimizing the false positive rate.

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[379]:


from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhatt
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


# In[380]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
m = MultiLabelBinarizer().fit(y_true)
f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')


# In[381]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')


# In[382]:


precision_recall_fscore_support(y_true, y_pred, average='micro')


# In[383]:


precision_recall_fscore_support(y_true, y_pred, average='weighted')


# In[384]:


precision_recall_fscore_support(y_true, y_pred)


# In[385]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


from sklearn.linear_model import LogisticRegression
clf= OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=600))
clf=clf.fit(X_train, y_train)


# In[25]:


yhattt = clf.predict(X_test)
print(yhatt)


# In[26]:


#Predict margin (libsvm name for this is predict_values)
y_scoreee = clf.decision_function(X_test)
#print(y_score)


# In[27]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scoreee[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[28]:


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scoreee.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[29]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[30]:


# Plot of a ROC curve for a specific class
#Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.
plt.figure()
plt.plot(fpr[5], tpr[5], label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[31]:


# Plot ROC curve
#the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one.
#This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
#The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate
#while minimizing the false positive rate.

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[32]:


from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
y_pred = yhattt
print (y_pred)
print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


# In[33]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
m = MultiLabelBinarizer().fit(y_true)
f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')


# In[34]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='macro')


# In[35]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='micro')


# In[36]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, average='weighted')


# In[37]:


precision_recall_fscore_support(y_true, y_pred)


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7 ]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[119]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


# In[ ]:





# In[120]:


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


# In[121]:


def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)))
    print("Hamming score: {}".format(hamming_score(y_pred, y_test)))
    print("---")    


# In[122]:


nb_clf = MultinomialNB()
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
lr = LogisticRegression(random_state=0, max_iter=900)
mn = LinearSVC(random_state=0,max_iter=130000, tol=1e-5)
prc = Perceptron(tol=1e-3, random_state=0)
bst = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
pag =PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-5)

for classifier in [nb_clf, sgd, lr, mn, prc,bst,pag]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_predd = clf.predict(X_test)
    print_score(y_predd, classifier)


# In[ ]:


# Hamming loss is the fraction of wrong labels to the total number of labels. In multi-class classification, hamming loss is calculated as the
# hamming distance between y_true and y_pred . In multi-label classification, hamming loss penalizes only the individual labels.


# In[ ]:


#Hamming Score is the fraction of correct predictions compared to the total labels. This is similar to Accuracy, and in fact they are interchangeable.
#Accuracy is simply the number of correct predictions divided by the total number of examples.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




