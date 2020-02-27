import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
import re
EPOCH=100
# spam_words=['call', 'free', 'txt', 'text', 'mobil', 'ur', 'stop', 'repli', 'prize', 'claim']
# ham_words=['go', 'get', 'nt', 'gt', 'lt', 'call', 'come', 'got', 'good', 'like']

x=pd.read_csv('./data/error_data/error.csv',encoding='latin-1')
x['error'][0]=re.sub('[¥]+', ' dollar ',x['error'][0])
print(x['error'][0])