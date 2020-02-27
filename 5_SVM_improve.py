from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import re
import torch.nn as nn
import torch
import scipy.io as scio
from sklearn import svm

EPOCH = 1000

# load the data
messages = pd.read_csv('./data/train.csv', encoding='latin-1')


# prosses the message(将句子分割成单词再处理)
def prosess_message(message):
    message = message.lower()
    message = re.sub('[¥]+', ' dollar ', message)  # 将¥变成英文字符
    word = word_tokenize(message)
    #  去除除字母外字符
    words = []
    for w in word:
        w = re.sub('[^a-z]', '', w)
        if w != '':
            words.append(w)
    # 去除长度小于2的单词
    words = [w for w in words if len(w) > 1]
    # 去除停用词
    sw = stopwords.words('english')
    words = [w for w in words if w not in sw]
    # 去除相同单词的不同形式
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return words


# prosess_message(messages['message'][0])
# print(messages['message'][0])

# 将句子处理成词向量
wordsList = []


def make_train_data():
    for i in range(len(messages['message'])):
        word = prosess_message(messages['message'][i])
        wordsList.append(word)


make_train_data()

# get the most common words in spam and ham
allwords = {'call': 11, 'dollar': 2, 'free': 4, 'txt': 6, 'text': 8, 'mobil': 10, 'ur': 12, 'stop': 14, 'repli': 16, 'prize': 18, 'claim': 20, 'go': 1, 'get': 3, 'nt': 5, 'gt': 7, 'lt': 9, 'come': 13, 'got': 15, 'good': 17, 'like': 19, 'ok': 21}


# make the feature matrix=(1,20)(对每封邮件找到其中是否出现过特征词，作为自己的特征矩阵)
train_data = []


def prosess_data(wordsList):
    for i in range(len(messages['message'])):
        train = torch.zeros(22)
        for j in range(len(wordsList[i])):
            if wordsList[i][j] in allwords:
                train[allwords[wordsList[i][j]]] = 1
        train_data.append(train)


prosess_data(wordsList)

label = messages['label'].values
label = torch.from_numpy(label).float()
train_data = torch.stack(train_data).to()
x_data = train_data.numpy()
y_data = label.numpy()
# print(x_data[0].shape)

clf = svm.SVC(C=0.5, kernel='linear')
clf.fit(x_data, y_data)


def accuracy(clf, x, y):
    predict_y = clf.predict(x)
    m = y.size
    count = 0
    for i in range(m):
        count = count + np.abs(int(predict_y[i]) - int(y[i]))  # 避免溢出错误得到225
    return 1 - float(count / m)


train_acc = accuracy(clf, x_data, y_data)
print('train acc for svm is: %.6f' % train_acc)

# test
# 将句子处理成词向量
test = pd.read_csv('./data/crossValidate.csv', encoding='latin-1')
testwords = []


def make_test_data():
    for i in range(len(test['message'])):
        word = prosess_message(test['message'][i])
        testwords.append(word)


make_test_data()

# make the feature matrix=(1,20)(对每封邮件找到其中是否出现过特征词，作为自己的特征矩阵)
test_data = []
def prosess_test_data():
    for i in range(len(test['message'])):
        k = torch.zeros(22)
        for j in range(len(testwords[i])):
            if testwords[i][j] in allwords:
                k[allwords[testwords[i][j]]] = 1
        test_data.append(k)
prosess_test_data()

test_label = test['label'].values
test_label = torch.from_numpy(test_label).float()
test_data = torch.stack(test_data).to()
x = test_data.numpy()
y = test_label.numpy()

test_acc = accuracy(clf, x, y)
print('test acc for svm is: %.6f' % test_acc)
