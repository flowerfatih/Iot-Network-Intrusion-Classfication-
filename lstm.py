# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:17:29 2022

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

#%% scaler for RNN model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(x_augmented)

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled,y,test_size = 0.3,random_state=42)

train = np.array(pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1))
test = np.array(pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1))

#%%

time_stemp = 100
dataX = []
dataY =  []

for i  in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp),0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
    
trainX = np.array(dataX)
trainY = np.array(dataY)

#%%
testX = []
testY =  []

for i  in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp),0]
    testX.append(a)
    testY.append(test[i + time_stemp, 0])
    
testx = np.array(testX)
testy = np.array(testY)
#%%
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
testx = np.reshape(testx, (testx.shape[0], testx.shape[1]))


#%%

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error

act_func = "softsign"

model = Sequential()
model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1,
                   return_sequences=True, activation=act_func))
model.add(LSTM(32, dropout=0.1, activation=act_func, return_sequences=True))
model.add(LSTM(32, dropout=0.1, activation=act_func))
model.add(Dense(128, activation=act_func))
model.add(Dropout(0.1))
model.add(Dense(256, activation=act_func))
model.add(Dropout(0.1))
model.add(Dense(128, activation=act_func))
model.add(Dropout(0.1))
model.add(Dense(1, name='out_layer', activation="linear"))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(trainX, trainY, epochs=50, batch_size=100)


#%%


















