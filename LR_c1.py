# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 21:00:53 2019

@author: aisun
"""
#回归方法，练习
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#sklearn一条一条数据 一条占一行

plt.figure(figsize=(16, 8))
plt.scatter(data['TV'], data['sales'], c ='black')
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()
#data数据不能够放在numpy里面直接用，需要转换成矩阵
#-1是可以自动变成一列的格式
x = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
#类的模型取名为reg
reg = LinearRegression()
#fit是训练和拟合
reg.fit(x, y)
#coef线性回归的系数.系数是矩阵
#intercept是偏置，偏置是向量
a = reg.coef_[0][0]
b = reg.intercept_[0]
#format用参变量的值代替格式符
print('a = {:.5}'.format(a))
print('b = {:.5}'.format(b))

print("线性模型为: Y = {:.5}X + {:.5} ".format(a, b))

pre = reg.predict(x)

plt.figure(figsize=(16,8))
plt.scatter(data['TV'], data['sales'], c ='black')
plt.plot(data['TV'], pre,c ='blue', linewidth=2)
plt.xlabel("Money spent on TV ads")
plt.ylabel("Sales")
plt.show()

#默认输入是矩阵
#predictions = reg.predict([[100]，[200],[300]])
predictions = reg.predict([[100]])
print('投入一亿元的电视广告, 预计的销售量为{:.5}亿'.format( predictions[0][0]) )
