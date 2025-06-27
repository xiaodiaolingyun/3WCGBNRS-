from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np



def compute_cart_ac(train_data, test_data):
    X_train = pd.DataFrame(train_data[:, :-1])
    y_train = pd.DataFrame(train_data[:, -1])
    X_test = pd.DataFrame(test_data[:, :-1])
    y_test = pd.DataFrame(test_data[:, -1])
    real = test_data[:, -1]

    # Replace NearestCentroid with DecisionTreeClassifier
    cart_model = DecisionTreeClassifier()
    cart_model.fit(X_train, y_train.values.ravel())
    predict = cart_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predict)
    precision = precision_score(y_test, predict)
    recall = recall_score(y_test, predict)
    f1 = f1_score(y_test, predict)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()

    # Additional metrics
    ur = len(predict[predict == -1]) / len(test_data)
    print("right_num, tp, fp, fn, tn:", accuracy, tp, fp, fn, tn)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Recall:", recall)
    print("Precision:", precision)
    return accuracy, f1, recall, precision


# filenames = [['banana(5300,2)','car(1728,6)','Dry_Bean_Dataset(13611,16)','elect(10000,13)','Energy Effciency(768,8)','HTRU(17000，8)','spambase(4601,57)'],
#              ['hcv(596,10)','mushroom(8124,22)'], ['heart(1025,13)'],['Endgame(958,9)'],['breast-cancer(699,9)']]

c_art_accuracy=[]
c_art_f1=[]
c_art_P=[]
c_art_R=[]
c_art_time=[]


# filenames = ['banana(5300,2)', 'car(1728,6)', 'Dry_Bean_Dataset(13611,16)', 'elect(10000,13)',
#              'Energy Effciency(768,8)', 'HTRU(17000，8)', 'spambase(4601,57)',
#              'hcv(596,10)', 'mushroom(8124,22)', 'heart(1025,13)', 'Endgame(958,9)', 'breast-cancer(699,9)']
filenames=['penbased(10992,16)']
for filename in filenames:
    # 调用函数进行十折交叉验证
    # 载入数据集
    df = pd.read_csv("datasets/" + filename + ".csv") # 替换为你的HCV数据集加载方式
    # 特征标准化
    da1 = df.values
    min_max = MinMaxScaler()
    da2 = min_max.fit_transform(da1[:, :-1])
    da3= np.hstack([da2, da1[:, -1:]])
    data = np.unique(da3, axis=0)
    # print()
    data_ml_knn = data[:, :-1]
    target_ml_knn = data[:, -1:]


    # 设置十折交叉验证
    k_fold = StratifiedKFold(n_splits=10)
    k_fold = k_fold.split(data, data[:, -1])
    cart_accuracy_list = []
    cart_f1_list = []
    cart_P_list = []
    cart_R_list = []
    cart_time_list = []
    # 进行十折交叉验证
    for k, (train_data_index, test_data_index) in enumerate(k_fold):
        print('迭代次数：{}'.format(k + 1))
        print('训练数据长度：{}'.format(len(train_data_index)))
        print('测试数据长度：{}'.format(len(test_data_index)))
        # 划分训练集和测试集
        train_data, test_data = data[train_data_index, :], data[test_data_index, :]


        # 调用函数进行训练和评估
        tic_0 = timer()
        cart_accuracy, cart_f1, cart_P, cart_R =compute_cart_ac(train_data, test_data,)
        tic_1 = timer()

        cart_time = tic_1 - tic_0
        cart_accuracy_list.append(cart_accuracy)
        cart_f1_list.append(cart_f1)
        cart_P_list.append(cart_P)
        cart_R_list.append(cart_R)
        cart_time_list.append(cart_time)
    c_art_accuracy.append(np.mean(cart_accuracy_list))
    c_art_f1.append (np.mean(cart_f1_list))
    c_art_P.append (np.mean(cart_P_list))
    c_art_R.append (np.mean(cart_R_list))
    c_art_time.append (np.mean(cart_time_list))
    data = {
            'CART': [c_art_accuracy, c_art_f1,c_art_R, c_art_P, c_art_time,],
            }
    index_names = ['Accuracy', 'F1', 'Re', 'P', 'Runtime']
    #  使用pandas创建DataFrame，并指定行名
    df = pd.DataFrame(data, index=index_names)
    # csv_name = 'test' +'-'+ filename + '.csv'
    # 输出表格到控制台
    print(df)
    # # 输出DataFrame为CSV文件
    df.to_csv('result/variousKNN-3WD/CART.csv', mode='a', index_label=filename, index=True)