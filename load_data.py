#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/11 21:46
# @Author  : FangXin

# 导入需要用到的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, -1].values

# 将数据分成训练集和测试集，训练集用于训练模型，测试集用于测试训练的模型的效果
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)

plt.figure(figsize=(10, 12))  # 设置画布大小
figure = plt.subplot(211)  # 将画布分成2行1列，当前位于第一块子画板

plt.scatter(X_train, Y_train, color='red')  # 描出训练集对应点
# plt.plot(X_train,predict,color='black')  #画出预测的模型图
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Train set')
plt.show()
