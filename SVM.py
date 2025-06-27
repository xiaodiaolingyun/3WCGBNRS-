
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from timeit import default_timer as timer


from sklearn.svm import SVC



def compute_svm_ac(train_data, test_data):
    X_train = pd.DataFrame(train_data[:, :-1])
    y_train = pd.DataFrame(train_data[:, -1])
    X_test = pd.DataFrame(test_data[:, :-1])
    y_test = pd.DataFrame(test_data[:, -1])
    real = test_data[:, -1]

    # Create an SVM model with default parameters
    svm_model = SVC()

    # Train the SVM model
    svm_model.fit(X_train, y_train.values.ravel())

    # Predict on the test set
    predict = svm_model.predict(X_test)

    # Performance metrics
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
    P = tp / (tp + fp + float("1e-8"))
    R = tp / (tp + fn + float("1e-8"))
    f1_score_val = (2 * P * R) / (P + R + float("1e-8"))

    print("right_num, tp, fp, fn, tn:", right_num, tp, fp, fn, tn)
    print(f'准确度: {accuracy}')
    print(f'查准率: {P}')
    print(f'查全率: {R}')
    print(f'F1评分: {f1_score_val}')
    return accuracy, f1_score_val, R, P

S_VM_accuracy=[]
S_VM_f1=[]
S_VM_P=[]
S_VM_R=[]
S_VM_time=[]

# filenames = ['banana(5300,2)', 'car(1728,6)', 'Dry_Bean_Dataset(13611,16)', 'elect(10000,13)',
#              'Energy Effciency(768,8)', 'HTRU(17000，8)', 'spambase(4601,57)',
#              'hcv(596,10)', 'mushroom(8124,22)', 'heart(1025,13)', 'Endgame(958,9)', 'breast-cancer(699,9)']

filenames=['penbased(10992,16)']
for filename in filenames:
    # 调用函数进行十折交叉验证
    # 载入数据集
    df = pd.read_csv("datasets/" + filename + ".csv")  # 替换为你的HCV数据集加载方式
    # 特征标准化
    da1 = df.values
    min_max = MinMaxScaler()
    da2 = min_max.fit_transform(da1[:, :-1])
    da3 = np.hstack([da2, da1[:, -1:]])
    data = np.unique(da3, axis=0)
    # print()
    data_ml_knn = data[:, :-1]
    target_ml_knn = data[:, -1:]

    # 设置十折交叉验证
    k_fold = StratifiedKFold(n_splits=10)
    k_fold = k_fold.split(data, data[:, -1])
    svm_accuracy_list = []
    svm_f1_list = []
    svm_P_list = []
    svm_R_list = []
    svm_time_list = []
    # 进行十折交叉验证
    for k, (train_data_index, test_data_index) in enumerate(k_fold):
        print('迭代次数：{}'.format(k + 1))
        print('训练数据长度：{}'.format(len(train_data_index)))
        print('测试数据长度：{}'.format(len(test_data_index)))
        # 划分训练集和测试集
        train_data, test_data = data[train_data_index, :], data[test_data_index, :]


        # 调用函数进行训练和评估
        tic_0 = timer()
        svm_accuracy, svm_f1, svm_P, svm_R = compute_svm_ac(train_data, test_data, )
        tic_1 = timer()

        svm_time = tic_1 - tic_0
        svm_accuracy_list.append(svm_accuracy)
        svm_f1_list.append(svm_f1)
        svm_P_list.append(svm_P)
        svm_R_list.append(svm_R)
        svm_time_list.append(svm_time)

    S_VM_accuracy.append(np.mean(svm_accuracy_list))
    S_VM_f1.append(np.mean(svm_f1_list))
    S_VM_P.append(np.mean(svm_P_list))
    S_VM_R.append(np.mean(svm_R_list))
    S_VM_time.append(np.mean(svm_time_list))
    data = {
        'SVM': [S_VM_accuracy, S_VM_f1, S_VM_R, S_VM_P, S_VM_time, ],
    }
    index_names = ['Accuracy', 'F1', 'Re', 'P', 'Runtime']
    #  使用pandas创建DataFrame，并指定行名
    df = pd.DataFrame(data, index=index_names)
    # csv_name = 'test' +'-'+ filename + '.csv'
    # 输出表格到控制台
    print(df)
    # # 输出DataFrame为CSV文件
    df.to_csv('result/variousKNN-3WD/SVM.csv', mode='a', index_label=filename, index=True)