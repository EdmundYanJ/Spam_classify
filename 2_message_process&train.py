from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import re
import torch.nn as nn
import torch
EPOCH=1000
FEATURES=24
#load the data
messages=pd.read_csv('./data/train.csv',encoding='latin-1')

#prosses the message(将句子分割成单词再处理)
def prosess_message(message):
    message=message.lower()
    #message=re.sub('[¥]+', ' dollar ',message) #将¥变成英文字符
    word=word_tokenize(message)
    #  去除除字母外字符
    words = []
    for w in word:
        w = re.sub('[^a-z]', '', w)
        if w != '':
            words.append(w)
    # 去除长度小于2的单词
    words=[w for w in words if len(w)>1]
    # 去除停用词
    sw=stopwords.words('english')
    words=[w for w in words if w not in sw]
    # 去除相同单词的不同形式
    stemmer=PorterStemmer()
    words=[stemmer.stem(w) for w in words]
    return words

#prosess_message(messages['message'][0])
#print(messages['message'][0])

#make dict(建立字典找出垃圾邮件和正常邮件出现最多的10个单词做特征)
spam_dict={}
ham_dict={}
wordsList=[]
def make_dict():
    for i in range(len(messages['message'])):
        word=prosess_message(messages['message'][i])
        wordsList.append(word)
        if messages['label'][i]:
            for i in range(len(word)):
                if word[i] in spam_dict:
                    spam_dict[word[i]]+=1
                else:
                    spam_dict[word[i]]=1
        else:
            for i in range(len(word)):
                if word[i] in ham_dict:
                    ham_dict[word[i]]+=1
                else:
                    ham_dict[word[i]]=1

make_dict()
#print(wordsList[0])

# get the most common words in spam and ham
spam_words={}
ham_words={}
def most_common_words():
    for i in range(int(FEATURES/2)):
        spam_words[max(spam_dict, key=spam_dict.get)]=2*i
        ham_words[max(ham_dict, key=ham_dict.get)]=2*i+1
        spam_dict.pop(max(spam_dict, key=spam_dict.get))
        ham_dict.pop(max(ham_dict, key=ham_dict.get))
most_common_words()
allwords={**spam_words,**ham_words}
print(ham_words)

# make the feature matrix=(1,20)(对每封邮件找到其中是否出现过特征词，作为自己的特征矩阵)
train_data=[]
def prosess_data(wordsList):
    for i in range(len(messages['message'])):
        train=torch.zeros(FEATURES)
        for j in range(len(wordsList[i])):
            if wordsList[i][j] in allwords:
                train[allwords[wordsList[i][j]]]=1
        train_data.append(train)
prosess_data(wordsList)
#print(train_data[0])
# data_column=pd.DataFrame({'features':train_data,'label':messages['label']})
# data_column.to_csv('./data/prosessed_data/Ptrain.csv')

#train the model(训练)
class LogisticRegretion(nn.Module):
    def __init__(self):
        super(LogisticRegretion,self).__init__()
        self.lr=nn.Linear(FEATURES,1)
        self.sm=nn.Sigmoid()

    def forward(self,x):
        x=self.lr(x)
        x=self.sm(x)
        return x

logistic_model=LogisticRegretion()
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(logistic_model.parameters())

label=messages['label'].values
label=torch.from_numpy(label).float()
train_data=torch.stack(train_data).to() #转换成torch训练格式
#print(label[0])
x_data=train_data
y_data=label
for epoch in range(EPOCH):
    out=logistic_model(x_data)
    loss=criterion(out,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%50==0:
        print_loss = loss.item()
        mask = out.ge(0.5).float()
        correct=0
        for i in range(len(messages['label'])):
            if mask[i]==messages['label'][i]:
                correct+=1
        acc=correct/len(messages['label'])
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度

#test
test=pd.read_csv('./data/crossValidate.csv',encoding='latin-1')
#print(len(test['label']))
test_words=[]
for i in range(len(test['message'])):
    word = prosess_message(test['message'][i])
    test_words.append(word)

test_data=[]
for i in range(len(test['message'])):
    k = torch.zeros(FEATURES)
    for j in range(len(test_words[i])):
        if test_words[i][j] in allwords:
            k[allwords[test_words[i][j]]] = 1
    test_data.append(k)


test_data=torch.stack(test_data).to() #转换成torch训练格式
out = logistic_model(test_data)
correct=0
mask = out.ge(0.5).float()
for i in range(len(test['label'])):
    if mask[i] == test['label'][i]:
        correct += 1
acc = correct / len(test['label'])
print('*' * 10)
print('test acc is {:.4f}'.format(acc))  # 精度

#查看分类错误数据，寻找优化方法
def LookForError():
    error_mes=[]
    out = logistic_model(x_data)
    mask = out.ge(0.5).float()
    hepothes =[]
    true=[]
    for i in range(len(messages['label'])):
        if mask[i] != messages['label'][i]:
            error_mes.append(messages['message'][i])
            hepothes.append(mask[i])
            true.append(messages['label'][i])
    data_column=pd.DataFrame({'error':error_mes,'hepothes':hepothes,'true':true})
    data_column.to_csv('./data/error_data/error.csv')
    print('save complete')
#LookForError()
