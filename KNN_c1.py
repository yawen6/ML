# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 03:17:07 2019

@author: aisun
"""


#使用sklearn自带数据集
from sklearn import datasets
#把数据分成训练集和测试集
from sklearn.model_selection import train_test_split
#sklearn在neighbors里面存在 KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#加载数据集，放到iris里面
#数据集来自UCI，为iris，里面有三类，0，1，2
iris = datasets.load_iris()

# 读取数据 X, y
#得到特征矩阵x,N*D.N:多少个样本，D：每个样本有多少个特征，D=4
#y为标签，是向量
X = iris.data
y = iris.target
print (X, y)


# 把数据分成训练数据和测试数据
#train_test_split分训练数据和测试数据，测试数据默认是0.25
#2003为随机种子，保证每次生成的测试数据是一致的
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

#构建KNN模型， K值为3、 并做训练
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)#训练过程

# 计算准确率
#from sklearn.metrics import accuracy_score
#clf.predict(X_test)针对测试数据做预测
#如果预测和真实的分类一样，就计数
correct = np.count_nonzero((clf.predict(X_test)==y_test)==True)
#自定义的计算准确率函数
#accuracy_score(y_test, clf.predict(X_test))
print ("Accuracy is: %.3f" %(correct/len(X_test)))

