#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/11 21:54
# @Author  : FangXin

# 导入需要用到的库
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

######################################################
# load data
######################################################
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, -1].values

# 将数据分成训练集和测试集，训练集用于训练模型，测试集用于测试训练的模型的效果
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

######################################################
# 最小二乘法
######################################################

# 根据公式求 w = (X'X)^(-1)X'Y  https://blog.csdn.net/weixin_49708196/article/details/120105526?spm=1001.2014.3001.5502
LAMBDA = 250
w = np.squeeze((np.linalg.inv(X_train.T @ X_train+LAMBDA) @ X_train.T @ Y_train))  # 250是l2正则系数
y_hat = np.mean(Y_train)
x_hat = np.mean(X_train)
# 计算b
b = y_hat - w * x_hat
print("w: ", w, "b: ", b)

predict = w * X_train + b


######################################################
# 画图
######################################################
plt.figure(figsize=(10, 12))  # 设置画布大小
figure = plt.subplot(211)  # 将画布分成2行1列，当前位于第一块子画板

plt.scatter(X_train, Y_train, color='red')  # 描出训练集对应点
plt.plot(X_train, predict, color='black')  # 画出预测的模型图
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Train set')

# 将模型应用于测试集，检验拟合结果
plt.subplot(212)
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, predict, color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Test set')
plt.show()

# 效果和 sklearn 类似
