# Spam_classify
## Dependencies
python 3.6

Pytorch 0.4.1

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
  - `numpy`
  - `pandas`
  - `nltk`
  - `re`
  - `sklearn`
## Data
Downloaded at kaggle.[dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)<br>
In this project you can use the data in the directory of data.
## Training&Test
- `python 1_data_process.py`
- `python 2_message_process&train.py`
- `python 3_SVM.py`
- `python 4_logistic_improve.py`
- `python 5_SVM_improve.py`<br>
其中，数据处理功能在1中完成，2和3为初步建立模型（2使用逻辑回归，3使用SVM），4和5分别是对2和3的优化。
## Result
![result](https://github.com/EdmundYanJ/Spam_classify/blob/master/result/Result.png)
