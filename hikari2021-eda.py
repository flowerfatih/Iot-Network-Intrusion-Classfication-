# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:39:03 2022

@author: fthsl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk as nlp
from collections import Counter

data = pd.read_csv("ALLFLOWMETER_HIKARI2021.csv")

# print(data.head())

# print(data.describe())

# print(data.columns)


# corr = data.corr()


#%%

print(data[["traffic_category","Label"]].groupby(["traffic_category"], as_index = False).mean().sort_values(by="Label",ascending = False))


print(data[["bwd_init_window_size","Label"]].groupby(["bwd_init_window_size"], as_index = False).mean().sort_values(by="Label",ascending = False))

#%%

def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        Q1 = np.percentile(df[c], 25)
        
        Q3 = np.percentile(df[c], 75)
        
        IQR = Q3 - Q1
        
        outlier_step = IQR * 1.5
        
        outlier_list_col= df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers
   


#%%




#%%
y = data['Label']

x_data = data.drop(['Label', 'Unnamed: 0.1', 'Unnamed: 0', 'uid', 'originh', 'responh', 'traffic_category' ], axis=1)
#%%
#x_data.traffic_category = [0 if each == "Benign" or each == "Background" else 1 for each in x_data.traffic_category]
description_list = []
for description in data.traffic_category:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()   # buyuk harftan kucuk harfe cevirme
    description = nlp.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)


#%%
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullandigim metot
max_features = 5000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

#%%
augment = np.array(sparce_matrix)
df = pd.DataFrame(augment)
x_augmented = pd.concat([x_data, df], axis=1)
#%%

print(x_augmented.info())

#object uid, originh responh=unique ip addresses

x = (x_augmented - np.min(x_augmented))/(np.max(x_augmented)-np.min(x_augmented))

#%% scaler for RNN model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(x_augmented)

#%%



#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

from sklearn.svm import SVC

svm = SVC(random_state=42)
svm.fit(x_train, y_train)

print("accuracy:{}".format(svm.score(x_test, y_test)))
#0.9328386879892426 ---------> traffic_category sütununu çıkarınca
#1.0                ---------> traffic_category sütununu nlp teknikleri ile sayıya çevirince
#%%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)




#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


#%%

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=lr, X=x_train, y=y_train, cv=10)

print("average accuracies: ", np.mean(accuracies))
print("average std: ", np.std(accuracies))



#%%

lr.fit(x_train, y_train)
print("accuracy:{}".format(lr.score(x_test,y_test)))
#0.9302243912980839 -------> traffic_category sütununu çıkarınca
#1.0                -------> traffic_category sütununu nlp teknikleri ile sayıya çevirince

#%%
from sklearn.model_selection import GridSearchCV
param_grid = {"C":np.logspace(-3, 3, 7), "penalty": ["l2", "l2"]} #l1 = lasso l2 = ridge

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(x,y)


print("tuned hyperparameter: ",logreg_cv.best_params_)
print("best score:", logreg_cv.best_score_)

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

logreg_cv = GridSearchCV(logreg, param_grid, cv=10)
logreg_cv.fit(x_train,y_train)

print("tuned hyperparameter: ",logreg_cv.best_params_)
print("best score:", logreg_cv.best_score_)

#%%

# c: 0.01, penalty:l2
logreg2 = LogisticRegression(C=0.01, penalty='l2')
logreg2.fit(x_train, y_train)
print("score: ", logreg2.score(x_test,y_test))






