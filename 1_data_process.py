import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log,sqrt
import pandas as pd
import numpy as np

#load data
mails=pd.read_csv('./data/raw/spam.csv',encoding='latin-1')
data_column=list(mails.columns)
mails=mails[[data_column[1],data_column[0]]]
mails.columns=['message','label']

#make the label y=(0,1)
for i in range(len(mails['label'])):
    if mails['label'][i]=='spam':
        mails['label'][i]=1
    else:
        mails['label'][i]=0
#print(mails.head())

#generate the train/cv/test dataset 6:2:2
trainIndex,remainIndex=[],[]
for i in range(len(mails['message'])):
    if np.random.uniform(0,1)<0.6:
        trainIndex.append(i)
    else:
        remainIndex.append(i)

crossValidate,testIndex=[],[]
for i in remainIndex:
    if np.random.uniform(0,1)<=0.5:
        crossValidate.append(i)
    else:
        testIndex.append(i)
trainData=mails.loc[trainIndex]
crossValidateData=mails.loc[crossValidate]
testData=mails.loc[testIndex]

trainData.reset_index(inplace=True)
trainData.drop(['index'],axis=1,inplace=True)
crossValidateData.reset_index(inplace=True)
crossValidateData.drop(['index'],axis=1,inplace=True)
testData.reset_index(inplace=True)
testData.drop(['index'],axis=1,inplace=True)

#save data
trainData.to_csv('./data/train.csv')
crossValidateData.to_csv('./data/crossValidate.csv')
testData.to_csv('./data/test.csv')
#print(trainData.head())
