import numpy as np
from sklearn.neighbors import NearestCentroid

from sklearn.datasets import load_iris

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_nc_ac(train_data, test_data):
    X_train = pd.DataFrame(train_data[:, :-1])
    # print(X_train)
    y_train = pd.DataFrame(train_data[:, -1])
    # print(y_train)
    X_test = pd.DataFrame(test_data[:, :-1])
    # print(X_test)
    y_test = pd.DataFrame(test_data[:, -1])
    real = test_data[:, -1]
    # print(real)
    nc_model = NearestCentroid()
    nc_model.fit(X_train, y_train.values.ravel())
    predict = nc_model.predict(X_test)
    # print(predict)
    # print(f"Model Classification Report :{classification_report(y_test, nc_model.predict(X_test))}")
    right_num = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    uc = 0
    for i in range(len(X_test)):
        if predict[i] == real[i]:
            right_num += 1
        if predict[i] == -1:
            uc += 1
        if real[i] == 1 and predict[i] == 1:
            tp += 1
        elif real[i] == 1 and predict[i] == 0:
            fn += 1
        elif real[i] == 0 and predict[i] == 1:
            fp += 1
        elif real[i] == 0 and predict[i] == 0:
            tn += 1
    accuracy = right_num / len(X_test)
    # 查准率，精确率 越高越好
    P = tp / (tp + fp + float("1e-8"))
    # 查全率，召回率 越高越好
    R = tp / (tp + fn + float("1e-8"))
    # f1评分，数值越大越稳定，（但还要考虑模型的泛化能力，不能造成过拟合）
    f1_score = (2 * P * R) / (P + R + float("1e-8"))
    # 边界域元素所占比例：
    ur = uc / len(test_data)
    # 代价参数：5 1 0.5
    # cost = 5 * (fn / len(test_data)) + 1 * (fp / len(test_data)) + 0.5 * ur
    print("right_num,tp,fp ,fn,tn:", right_num, tp, fp, fn, tn)
    return accuracy, f1_score, R,P,

