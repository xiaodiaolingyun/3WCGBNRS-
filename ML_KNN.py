# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:31:16 2017

@author: 14094
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

# import sampleProcess as sp
from sklearn.model_selection import KFold, StratifiedKFold
from ML_KNN_tools import knn1,knn


class ML_KNN(object):
    s = 1
    k = 10
    labels_num = 0
    train_data_num = 0
    train_data = np.array([])
    train_target = np.array([])
    # test_data = np.array([])
    # test_target = np.array([])
    rtl = np.array([])
    Ph1 = np.array([])  # P(H1)
    Ph0 = np.array([])
    Peh1 = np.array([])
    Peh0 = np.array([])
    predict_labels = np.array([])

    def __init__(self, _train_data, _train_target, _k):
        self.train_data = _train_data
        self.train_target = _train_target
        self.k = _k
        self.labels_num = _train_target.shape[1]
        self.train_data_num = self.train_data.shape[0]
        self.Ph1 = np.zeros((self.labels_num,))
        self.Ph0 = np.zeros((self.labels_num,))
        self.Peh1 = np.zeros((self.labels_num, self.k + 1))
        self.Peh0 = np.zeros((self.labels_num, self.k + 1))

    def fit(self):
        for i in range(self.labels_num):
            y = 0
            for j in range(self.train_data_num):
                if self.train_target[j][i] == 1:
                    y = y + 1
            self.Ph1[i] = (self.s + y) / (self.s * 2 + self.train_data_num)
        self.Ph0 = 1 - self.Ph1

        for i in range(self.labels_num):
            c1 = np.zeros((self.k + 1,))
            c0 = np.zeros((self.k + 1,))
            for j in range(self.train_data_num):
                temp = 0
                KNN = knn(self.train_data, j, self.k)
                for k in range(self.k):
                    if self.train_target[int(KNN[k])][i] == 1:
                        temp = temp + 1
                if self.train_target[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1

            for l in range(self.k + 1):
                self.Peh1[i][l] = (self.s + c1[l]) / (self.s * (self.k + 1) + c1.sum())
                self.Peh0[i][l] = (self.s + c0[l]) / (self.s * (self.k + 1) + c0.sum())

    def predict(self, _test_data):
        self.rtl = np.zeros((_test_data.shape[0], self.labels_num))
        test_data_num = _test_data.shape[0]
        self.predict_labels = np.zeros((test_data_num, self.labels_num))
        for i in range(test_data_num):
            KNN = knn1(self.train_data, _test_data[i], self.k)
            for j in range(self.labels_num):
                temp = 0
                y1 = 0
                y0 = 0
                for k in range(self.k):
                    if self.train_target[int(KNN[k])][j] == 1:
                        temp = temp + 1
                y1 = self.Ph1[j] * self.Peh1[j][temp]
                y0 = self.Ph0[j] * self.Peh0[j][temp]
                self.rtl[i][j] = self.Ph1[j] * self.Peh1[j][temp] / (
                        self.Ph1[j] * self.Peh1[j][temp] + self.Ph0[j] * self.Peh0[j][temp])
                if y1 > y0:
                    self.predict_labels[i][j] = 1
                else:
                    self.predict_labels[i][j] = 0
        # print(self.predict_labels)
        return self.predict_labels


def compute_mlknn_ac(data, target, tr_index, val_index,k_value):
    tr_X, val_X = data[tr_index], data[val_index]
    tr_Y, val_Y = target[tr_index], target[val_index]
    mlKnn = ML_KNN(tr_X, tr_Y, k_value)
    mlKnn.fit()
    labels1 = mlKnn.predict(val_X)
    predict = labels1.flatten()
    real = val_Y.flatten()
    right_num = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    uc = 0
    for i in range(len(labels1)):
        if predict[i] == real[i]:
            right_num += 1
        if real[i] == 1 and predict[i] == 1:
            tp += 1
        elif real[i] == 1 and predict[i] == 0:
            fn += 1
        elif real[i] == 0 and predict[i] == 1:
            fp += 1
        elif real[i] == 0 and predict[i] == 0:
            tn += 1
    accuracy = right_num / len(val_index)
    # 查准率，精确率 越高越好
    P = tp / (tp + fp + float("1e-8"))
    # 查全率，召回率 越高越好
    R = tp / (tp + fn + float("1e-8"))
    # f1评分，数值越大越稳定，（但还要考虑模型的泛化能力，不能造成过拟合）
    f1_score = (2 * P * R) / (P + R + float("1e-8"))
    # # 边界域元素所占比例：
    # ur = uc / len(val_index)
    # # 代价参数：5 1 0.5
    # cost = 5 * (fn / len(val_index)) + 1 * (fp / len(val_index)) + 0.5 * ur
    print("right_num,tp,fp ,fn,tn:",right_num,tp,fp ,fn,tn)
    return accuracy, f1_score, P, R,
