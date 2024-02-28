#!/usr/bin/env python
# coding: utf-8

# In[276]:


name_string = '20210514'
# https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff


# In[277]:


import matplotlib.pyplot as plt


# In[278]:


import pandas as pd
import pyodbc

# Environment settings: 
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[279]:


# set up connection to the SQL server
conn = pyodbc.connect('Driver={SQL Server};'
'Server=MCS-TEST-SQL;'
'Database=MCS;'
'Trusted_Connection=yes;')


# In[280]:


# SQL query
defaults_sql = '''
SELECT * FROM Dataset

'''


# In[281]:


# execute query and save results to Pandas dataframe
defaults = pd.read_sql(defaults_sql.format('[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]', '[MCS].[NT0001\BF4510].[Dataset]'), conn)
defaults.shape


# In[282]:


# SQL query for test set
defaults_sql_test = '''
SELECT * FROM Dataset_test

'''


# In[283]:


# execute query and save results to Pandas dataframe
defaults_test = pd.read_sql(defaults_sql_test.format('[MCS].[NT0001\BF4510].[Dataset_test]', '[MCS].[NT0001\BF4510].[Dataset_test]', '[MCS].[NT0001\BF4510].[Dataset_test]', '[MCS].[NT0001\BF4510].[Dataset_test]'), conn)
defaults_test.shape


# In[ ]:





# In[284]:


# and export it as a csv
defaults.to_csv('base_table_' + name_string + '.csv', index=False)


# In[285]:


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


# In[286]:


# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[287]:


# get the dataset
def get_dataset():
	X = defaults.drop(columns = ['vie','du','tr','ket' ,'pen','ses','sep' ,'ast'])
	X.shape
	y = defaults.drop(columns = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay'
	,'Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])
	y.shape
	return X, y





# In[288]:


# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(12, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(8, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(8, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model


# In[289]:


# evaluate a model using repeated k-fold cross-validation
# https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	print(n_inputs, n_outputs)
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
        
		print(X_train, X_test)
		print(y_train, X_test)
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=600)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
        
        #cm=multilabel_confusion_matrix()
        #print(cm)
	return results


# In[290]:


# load dataset
X, y = get_dataset()
# class_names = ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential, 'global']
               
                        


# In[291]:


X.describe()


# In[292]:


y.describe()


# In[298]:


# create dataframe from file
dataframe = y

# use corr() method on dataframe to
# make correlation matrix
matrix = dataframe.corr()
 
# print correlation matrix
print("Correlation Matrix is : ")
print(matrix)

corr = dataframe.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps


# In[299]:


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


# In[ ]:





# In[300]:


import matplotlib.pyplot as plt

plt.matshow(dataframe.corr())
plt.show()


# In[296]:


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


# In[297]:


n_inputs, n_outputs = X.shape[1], y.shape[1]
print(n_inputs, n_outputs)
# get model
model = get_model(n_inputs, n_outputs)


# In[221]:


# evaluate model
results = evaluate_model(X.values, y.values)


# In[222]:


# summarize performance
#accuracy = number of correct predictions / total predictions
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))


# In[223]:


# https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-confusion-matrix


# In[224]:


get_ipython().system('pip install imblearn  ')


# In[225]:


from imblearn.pipeline import make_pipeline
from collections import Counter


# In[226]:


# make a pipeline without resampling for validation and get predictions
# The 0.5 threshold suggestion is for sigmoid function, because it is symmetric around 0 and hits 0.5 at 0
# should also tune your threshold. Several statistics such as ROC curve, Precision/Recall curves are obtained the measure 
# the performance while changing this threshold, and they're used to understand the behavior of the system.
#  more commonly suggested option for sigmoid, for instance, is to use your class priors.

# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

# many machine learning algorithms are capable of predicting a probability or scoring of class membership, and this must be
#interpreted before it can be mapped to a crisp class label. This is achieved by using a threshold, such as 0.5, where 
#all values equal or greater than the threshold are mapped to one class and all other values are mapped to another class.
#For those classification problems that have a severe class imbalance, the default threshold can result in poor performance.
#As such, a simple and straightforward approach to improving the performance of a classifier that predicts probabilities 
# on an imbalanced classification problem is to tune the threshold used to map probabilities to class labels.

# Label cardinality signifies the average number of labels present in the training data set. Label cardinality is 
#independent of the number of labels present in the dataset. Label density takes into consideration the number of labels
#present in the dataset.

#Thresholding strategy:
#Calibrate a threshold such that LCard(Y) ≈ LCard(Yˆ )
#I e.g., training data has label cardinality of 1.7;
#I set a threshold t such that the label cardinality of the test data is as
#close as possible to 1.7

pipeline = make_pipeline( model)
predictions = model.predict(X.values)
print(predictions)


# In[227]:


#I prefer to leave the threshold at 0.5 for one simple reason: In classification you are trying to train a model which can 
#clearly separate between the classes/labels, i.e. for binary case predict 0 and 1 as good as possible.
#So if your model is equally good in predicting 0’s as it is in predicting 1’s you will have many predicted probabilities
#close to 0 and many close to 1 and thus a broad margin between the two classes in the middle. So a threshold of 0.5 will
#nicely work in this case.
#However, if the model is not good in predicting one or both classes the probabilities will be spread out more between 0.0 
#and 1.0. Then you could try to play with the threshold or you can try to improve you model to be able to better separate 
#etween both classes. Same applies for Multi-label classification as it is also using binary cross-entropy and each label
#is predicted with a probability between 0 and 1.


# In[228]:


# https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la
# https://stackoverflow.com/questions/38015181/accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target/54458777
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
#In multilabel confusion matrix , the count of true negatives, false negatives, true positives and false positives
# https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/Multi_Label_Model_Evaulation.ipynb
# iš esmės neteisingai lyginama - reikia keisti modelį į klasifikacijos, o ne tikimybinį

from sklearn.metrics import multilabel_confusion_matrix

y_pred=predictions
print(y_pred)
#y_pred=np.argmax(y_pred, axis=1)
y_pred=y_pred.round()
print(y_pred)
#yy = np.array(y)
#print(yy)
y_test=y
print(y_test)
cm = multilabel_confusion_matrix(y_test, y_pred)
print(cm)


# In[230]:


from sklearn.metrics import classification_report
label_names = ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential', 'global']
print(classification_report(y.values, predictionsfinalr,target_names=label_names))


# In[231]:


from sklearn.metrics import hamming_loss, accuracy_score 
y_true = y_test
print (y_true)
#y_pred = yhat
print (y_pred)
#print("accuracy_score:", accuracy_score(y_true, y_pred))
print("Hamming_loss:", hamming_loss(y_true, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
y_test.reset_index(drop=True)
model.fit(X_train.values, y_train.values, verbose=0, epochs=400)
y_pred = model.predict(X_test.values)
y_pred=y_pred.round()
print(y_pred)
#yy= np.array(y_test)
print(np.array(y_test))

cm = multilabel_confusion_matrix(np.array(y_test), y_pred)
print(cm)

# https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
# https://www.slideshare.net/SridharNomula1/evaluation-of-multilabel-multi-class-classification-147091418


# In[187]:


from sklearn import metrics
from sklearn.metrics import recall_score, f1_score, precision_score
recall_score(y_true=np.array(y_test), y_pred=y_pred, average='weighted')


# In[188]:


# This means that there is no F-score to calculate for this label, and thus the F-score for this case is considered
# to be 0.0. Since you requested an average of the score, you must take into account that a score of 0 was included in the
# calculation, and this is why scikit-learn is showing you that warning
# When true positive + false positive == 0, precision is undefined; When true positive + false negative == 0, 
# recall is undefined. In such cases, by default the metric will be set to 0, as will f-score, and UndefinedMetricWarning 
# will be raised. This behavior can be modified with zero_division


# In[189]:


# This error occurs under two circumstances:
#If you have used train_test_split() to split your data, you have to make sure that you reset the index of the data
#(specially when taken using a pandas series object): y_train, y_test indices should be resetted. 
#The problem is when you try to use one of the scores from sklearn.metrics such as; precision_score,
#this will try to match the shuffled indices of the y_test that you got from train_test_split().
#so use, either np.array(y_test) for y_true in scores or y_test.reset_index(drop=True)

#Then again you can still have this error if your predicted 'True Positives' is 0, which is used for precision,
#recall and f1_scores. You can visualize this using a confusion_matrix. If the classification is multilabel and you set
#param: average='weighted'/micro/macro you will get an answer as long as the diagonal line in the matrix is not 0


# In[190]:


# The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value 
# at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for 
# the F1 score is:
# F1 = 2 * (precision * recall) / (precision + recall)
# In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending
# on the average parameter.
# average='weighted': Calculate metrics for each label, and find their average weighted by support (the number of true
# instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not
# between precision and recall.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#  all metrics for classification: https://scikit-learn.org/stable/modules/model_evaluation.html


# In[191]:


precision_score(y_true=np.array(y_test), y_pred=y_pred, average='weighted')


# In[192]:



f1_score(y_true=np.array(y_test), y_pred=y_pred, average='weighted')


# In[193]:


f1_score(y_true=np.array(y_test), y_pred=y_pred, average=None)


# In[194]:


recall_score(y_true=np.array(y_test), y_pred=y_pred, average=None)


# In[195]:


precision_score(y_true=np.array(y_test), y_pred=y_pred, average=None)


# In[196]:


# The AUC value lies between 0.5 to 1 where 0.5 denotes a bad classifer and 1 denotes an excellent classifier


# In[197]:


from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
print (np.array(y_test))
for i in range(8):
    fpr[i], tpr[i], _ = roc_curve( np.array(y_test)[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[ ]:





# In[198]:


# fit the model on all data
model.fit(X.values, y.values, verbose=0, epochs=600)


# In[199]:


predictionsfinal = model.predict(X.values)
print(predictionsfinal)


# In[71]:


# Final evaluation


# In[72]:


predictionsfinalr=predictionsfinal.round()
recall_score(y_true=np.array(y), y_pred=predictionsfinalr, average='weighted')


# In[73]:


precision_score(y_true=np.array(y), y_pred=predictionsfinalr, average='weighted')


# In[74]:


f1_score(y_true=np.array(y), y_pred=predictionsfinalr, average='weighted')


# In[75]:


recall_score(y_true=np.array(y), y_pred=predictionsfinalr, average=None)


# In[76]:


precision_score(y_true=np.array(y), y_pred=predictionsfinalr, average=None)


# In[77]:


f1_score(y_true=np.array(y), y_pred=predictionsfinalr, average=None)


# In[78]:


# Compute micro-average ROC curve and ROC area
from sklearn.metrics import roc_curve
roc_auc_score(y.values, predictionsfinal, average=None)


# In[79]:


from sklearn.metrics import classification_report
label_names = ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential', 'global']
print(classification_report(y.values, predictionsfinalr,target_names=label_names))
#Support is the number of actual occurrences of the class in the specified dataset. 
#Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and
#could indicate the need for stratified sampling or rebalancing.

#precision is the ability of the classifier not to label as positive a sample that is negative,
#and recall is the ability of the classifier to find all the positive samples.
#The F-measure can be interpreted as a weighted harmonic mean of the precision and recall:
#reaches its best value at 1 and its worst score at 0.


# In[80]:


#In multiclass and multilabel classification task, the notions of precision, recall, and F-measures can be applied
#to each label independently.
#There are a few ways to combine results across labels, specified by the average argument
#to the average_precision_score (multilabel only), f1_score, fbeta_score, precision_recall_fscore_support,
#precision_score and recall_score functions, as described above.
#Note that for “micro”-averaging in a multiclass setting with all labels included will produce equal precision,
# recall and F, while “weighted” averaging may produce an F-score that is not between precision and recall.

# https://colab.research.google.com/github/kmkarakaya/ML_tutorials/blob/master/Multi_Label_Model_Evaulation.ipynb#scrollTo=v9Em4uWRTCf-
# average parameter is required for multiclass/multilabel targets.
#  None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for
# each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision
# and recall.
# 'samples': Calculate metrics for each instance, and find their average (only meaningful for multilabel classification 
# where this differs from accuracy_score).


# In[ ]:





# In[81]:


# https://sites.google.com/site/nttrungmtwiki/home/it/data-science---python/multiclass-and-multilabel-roc-curve-plotting
# https://scikit-learn.org/0.15/auto_examples/plot_roc.html
#ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. This means that 
#the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. 
#This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
#The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.
#ROC curves are typically used in binary classification to study the output of a classifier. In order to extend ROC curve and ROC area to multi-class or multi-label classification, it is necessary to binarize the output. One ROC curve can be drawn per label, but one can also draw a ROC curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).


# In[ ]:





# In[82]:


# make a prediction for new data
row = [5, 12, 15, 20, 5, 5, 1, 7, 6, 2, 6, 17]
print(row)
newX = np.asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])


# In[83]:


# make a prediction for new data
row = [1, 2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]
print(row)
newX = np.asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[232]:


#Calculate shapley values


# In[233]:


import shap
import ipywidgets
from ipywidgets import IntProgress


# In[234]:


# Need to load JS vis in the notebook
shap.initjs()


# In[235]:


# Here we use a selection of 99 samples from the dataset to represent “typical” feature values,
# and then use 100 perterbation samples to estimate the SHAP values for a given prediction. 
# Note that this requires 100 * 99 evaluations of the model.
# https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d


# In[236]:


explainer = shap.KernelExplainer(model.predict,X)


# In[237]:


print(X.shape)


# In[238]:


shap_values = explainer.shap_values(X)


# In[239]:


# At the end, we get a (n_samples,n_features) numpy array. Each element is the shap value of that feature of that record.
# Remember that shap values are calculated for each feature and for each record.
print (shap_values )


# In[240]:


pd.DataFrame(shap_values[0])


# In[241]:


pd.DataFrame(shap_values[1])


# In[242]:


pd.DataFrame(shap_values[2])


# In[243]:


pd.DataFrame(shap_values[3])


# In[244]:


pd.DataFrame(shap_values[4])


# In[245]:


pd.DataFrame(shap_values[5])


# In[246]:


pd.DataFrame(shap_values[6])


# In[247]:




pd.DataFrame(shap_values[7])


# In[248]:


pd.DataFrame(shap_values[7][0,:])


# In[249]:


pd.DataFrame(shap_values[7]).head()


# In[ ]:





# In[250]:


# Shap values show how much a given feature changed our prediction (compared to if we made that prediction at some baseline value (pradinė reikšmė) of that feature)
shap.summary_plot(shap_values,X_test,feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay'
	,'Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'],class_names= ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential', 'global'])


# In[251]:


#The SHAP value plot can further show the positive and negative relationships of the predictors with the target variable. 
shap.summary_plot(shap_values, X, feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'],class_names= ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential', 'global'])


# In[ ]:





# In[252]:


#summary_plot of a specific class for X
shap.summary_plot(shap_values, X.values, feature_names = X.columns)


# In[253]:


shap.summary_plot(shap_values[0], X.values, feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'],class_names= ['sensing', 'intuitive', 'visual', 'verbal', 'active', 'reflective', 'sequential', 'global'])


# In[254]:


#summary_plot of a specific class for X
shap.summary_plot(shap_values[1], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[2], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[3], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[4], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[5], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[6], X.values, feature_names = X.columns)
shap.summary_plot(shap_values[7], X.values, feature_names = X.columns)


# In[255]:


#summary_plot of a specific class for X_test
shap.summary_plot(shap_values, X.values, feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[256]:


#shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0,:])
#shap.force_plot(explainer.expected_value, shap_values, X_test)
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Force%20Plot%20Colors.html
#visualize prediction explanations 
shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[ ]:





# In[257]:


shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[258]:


shap.force_plot(explainer.expected_value[2], shap_values[2], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[259]:


shap.force_plot(explainer.expected_value[2], shap_values[2], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[260]:


shap.force_plot(explainer.expected_value[3], shap_values[3], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[261]:


shap.force_plot(explainer.expected_value[4], shap_values[4], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[262]:


shap.force_plot(explainer.expected_value[5], shap_values[5], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[263]:


shap.force_plot(explainer.expected_value[6], shap_values[6], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[264]:


shap.force_plot(explainer.expected_value[7], shap_values[7], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[265]:


shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[266]:


shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[267]:


shap.force_plot(explainer.expected_value[2], shap_values[2][0,:], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[268]:


shap.force_plot(explainer.expected_value[3], shap_values[3][0,:],  feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[269]:


shap.force_plot(explainer.expected_value[4], shap_values[4][0,:],  feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[270]:


shap.force_plot(explainer.expected_value[5], shap_values[5][0,:], feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[271]:


shap.force_plot(explainer.expected_value[6], shap_values[6][0,:],  feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[272]:


shap.force_plot(explainer.expected_value[6], shap_values[6][0,:],  feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[273]:


shap.force_plot(explainer.expected_value[7], shap_values[7][0,:],  feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[274]:


# The output value is the prediction for that observation (the prediction of the first row in Table B is 6.20).
# The base value: The original paper explains that the base value E(y_hat) is “the value that would be predicted if we
# did not know any features for the current output.” In other words, it is the mean prediction, or mean(yhat). 
# You may wonder why it is 5.62. This is because the mean prediction of Y_test is 5.62. 
# You can test it out by Y_test.mean() which produces 5.62.
# Red/blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.


# In[275]:


# make a prediction for new data
row = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1]
print(row)
newX = np.asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])


# In[135]:


#https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
shap_values = explainer.shap_values(newX)


# In[136]:


print(shap_values)


# In[137]:


pd.DataFrame(shap_values[0])


# In[138]:


pd.DataFrame(shap_values[1])


# In[139]:


pd.DataFrame(shap_values[2])


# In[140]:


pd.DataFrame(shap_values[3])


# In[141]:


pd.DataFrame(shap_values[4])


# In[142]:


pd.DataFrame(shap_values[5])


# In[143]:


pd.DataFrame(shap_values[6])


# In[144]:


pd.DataFrame(shap_values[7])


# In[145]:


shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[146]:


# make a prediction for new data
row = [10, 0, 1, 5, 9, 12, 10, 1, 1, 1,1, 1]
print(row)
newX = np.asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])


# In[147]:


shap_values = explainer.shap_values(newX)


# In[148]:


pd.DataFrame(shap_values[0])


# In[149]:


pd.DataFrame(shap_values[1])


# In[150]:


# shap.force_plot() takes three values: the base value (explainerModel.expected_value[0]), 
#???the SHAP values (shap_values_Model[j][0]) and the matrix of feature values (S.iloc[[j]]).
#The base value or the expected value is the average of the model output over the training data X. 
#It is the base value used in the following plot.
#Red/blue: Features that push the prediction higher (to the right) are shown in red, 
# and those pushing the prediction lower are in blue.
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[151]:


shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[152]:


shap.force_plot(explainer.expected_value[2], shap_values[2][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[153]:


shap.force_plot(explainer.expected_value[3], shap_values[3][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[154]:


shap.force_plot(explainer.expected_value[4], shap_values[4][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[155]:


shap.force_plot(explainer.expected_value[5], shap_values[5][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[156]:


shap.force_plot(explainer.expected_value[6], shap_values[6][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[157]:


shap.force_plot(explainer.expected_value[7], shap_values[7][0,:],feature_names = ['Navigation_deep','Navigation_skip_overview','Forum_visit','Forum_post','Video_pictures' ,'Content_text_stay','Feedback_no','NO_connections_links','Quiz_revisions','Ques_detail','Ques_facts','Ques_concepts'])


# In[ ]:





# In[ ]:





# In[ ]:




