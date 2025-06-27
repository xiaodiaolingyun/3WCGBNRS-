
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold
from timeit import default_timer as timer
# 数据集读取
from sklearn.preprocessing import MinMaxScaler
from tool import get_membership_table,compute_best_beta
from improved_GB import DBGBList
from original_GB import GBList
from tools import compute_ac
from NC import compute_nc_ac
from ML_KNN import compute_mlknn_ac




KNN_accuracy=[]
GB_accuracy_k =[]
DBGB_accuracy_k =[]
GB_accuracy = []
DBGB_accuracy = []
GB_H_accuracy=[]
DBGB_H_accuracy=[]
ml_knn_accuracy=[]
n_c_accuracy=[]

KNN_f1=[]
GB_f1_k=[]
DBGB_f1_k=[]
GB_f1 = []
DBGB_f1 = []
GB_H_f1= []
DBGB_H_f1= []
ml_knn_f1=[]
n_c_f1=[]

KNN_P=[]
GB_P_k=[]
DBGB_P_k=[]
GB_P = []
DBGB_P = []
GB_H_P= []
DBGB_H_P= []
ml_knn_P=[]
n_c_P=[]

KNN_R=[]
GB_R_k=[]
DBGB_R_k=[]
GB_R = []
DBGB_R = []
GB_H_R= []
DBGB_H_R= []
ml_knn_R=[]
n_c_R=[]

KNN_runtime=[]
GB_runtime_k=[]
DBGB_runtime_k=[]
GB_runtime = []
DBGB_runtime = []
GB_H_runtime= []
DBGB_H_runtime= []
ml_knn_runtime=[]
n_c_runtime=[]



filenames=['hcv(596,10).csv',]
for filename in filenames:
    knn_k = 1
    '''数据读取与预处理'''
    # 效果比较好的数据集
    df = pd.read_csv("datasets/" + filename  )
    print("———————————————————————————正在处理数据集:" + filename + "—————————————————————————————————————————————")
    # data = df.values
    '''归一化数据集'''
    da1 = df.values
    da2 = np.unique(da1, axis=0)
    min_max = MinMaxScaler()
    da3 = min_max.fit_transform(da2[:, :-1])
    # da3 = np.unique(da2, axis=0)
    data= np.hstack([da3, da2[:, -1:]])

    # print()
    data_ml_knn = data[:, :-1]
    target_ml_knn = data[:, -1:]#data[:, -1:] 返回一个二维子数组，而 data[:, -1] 返回一个一维数组。
    '''十折交叉验证'''             #如果您需要保留维度信息，可以使用 data[:, -1:]。
    # 原始十折交叉验证
    k_fold = StratifiedKFold(n_splits=10)
    k_fold = k_fold.split(data, data[:, -1])
    # 随机排列交叉验证
    # k_fold = ShuffleSplit(n_splits=10)
    # k_fold = k_fold.split(data)

    traditional_accuracy_list=[]
    old_accuracy_list=[]
    new_accuracy_list = []
    original_accuracy_list = []
    improved_accuracy_list = []
    original_H_accuracy_list=[]
    improved_H_accuracy_list=[]
    ml_knn_accuracy_list = []
    nc_accuracy_list = []

    traditional_f1_list=[]
    old_f1_list=[]
    new_f1_list = []
    original_f1_list = []
    improved_f1_list = []
    original_H_f1_list = []
    improved_H_f1_list = []
    ml_knn_f1_list = []
    nc_f1_list = []

    traditional_P_list = []
    old_P_list = []
    new_P_list = []
    original_P_list = []
    improved_P_list = []
    original_H_P_list = []
    improved_H_P_list = []
    ml_knn_P_list = []
    nc_P_list = []

    traditional_R_list = []
    old_R_list = []
    new_R_list = []
    original_R_list = []
    improved_R_list = []
    original_H_R_list = []
    improved_H_R_list = []
    ml_knn_R_list = []
    nc_R_list = []


    old_runtime_list = []
    new_runtime_list = []
    original_runtime_list = []
    improved_runtime_list = []
    original_H_runtime_list = []
    improved_H_runtime_list = []

    traditional_time_list= []
    old_time_list = []
    new_time_list = []
    original_time_list=[]
    improved_time_list=[]
    original_H_time_list = []
    improved_H_time_list = []
    ml_knn_time_list=[]
    nc_time_list=[]


    for k, (train_data_index, test_data_index) in enumerate(k_fold):
        # if k == 0:
        print('迭代次数：{}'.format(k + 1))
        print('训练数据长度：{}'.format(len(train_data_index)))
        print('测试数据长度：{}'.format(len(test_data_index)))
        train_data = data[train_data_index, :]
        test_data = data[test_data_index, :]
        # 使用训练数据生成粒球
        '''step 1 生成原始粒球列表'''
        granular_balls_original = GBList(train_data)  # create the list of granular balls
        granular_balls_original.init_granular_balls()  # initialize the list
        # 是否删除元素小于2的粒球对结果影响很大
        granular_balls_original.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample
        print('原始粒球数：{}'.format(len(granular_balls_original.granular_balls)))
        membership_list_original = get_membership_table(granular_balls_original)
        best_beta_original, best_alpha_original, min_fuzziness_original = compute_best_beta (granular_balls_original,membership_list_original)


        '''step 2 生成密度粒球列表'''
        granular_balls_improved = DBGBList(granular_balls_original)  # create the list of granular balls
        granular_balls_improved.init_granular_balls()  # initialize the list
        granular_balls_improved.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample
        print('DBSCAN改进后粒球数：{}'.format(len(granular_balls_improved.granular_balls)))
        membership_list_improved= get_membership_table(granular_balls_improved)
        best_beta_improved , best_alpha_improved , min_fuzziness_improved  = compute_best_beta(granular_balls_improved, membership_list_improved)



        # 使用测试集计算准确率
        tic_0 = timer()
        traditional_accuracy, traditional_f1, traditional_P, traditional_R = compute_ac(train_data, test_data, "T",
                                                                                       knn_k,
                                                                                        granular_balls_original,
                                                                                        granular_balls_improved,best_beta_improved , best_alpha_improved,best_beta_original,best_alpha_original)
        tic_1 = timer()
        old_accuracy, old_f1, old_P, old_R = compute_ac(train_data,test_data,"OK",knn_k,granular_balls_original,granular_balls_improved, best_beta_improved , best_alpha_improved,best_beta_original,best_alpha_original)
        tic_2 = timer()
        new_accuracy, new_f1, new_P, new_R= compute_ac(train_data,test_data, "NK",knn_k,granular_balls_original,granular_balls_improved, best_beta_improved , best_alpha_improved,best_beta_original,best_alpha_original)
        tic_3= timer()
        original_accuracy, original_f1, original_P, original_R = compute_ac(train_data,test_data, "O3",knn_k, granular_balls_original,granular_balls_improved, best_beta_original,best_alpha_original, best_beta_improved,best_alpha_improved)
        tic_4 = timer()
        improved_accuracy, improved_f1, improved_P, improved_R=compute_ac(train_data,test_data, "N3",knn_k,granular_balls_original,granular_balls_improved, best_beta_original, best_alpha_original,best_beta_improved,best_alpha_improved)
        tic_5 = timer()
        original_H_accuracy, original_H_f1, original_H_P, original_H_R = compute_ac(train_data,test_data, "OH",knn_k,  granular_balls_original,granular_balls_improved, best_beta_original,best_alpha_original, best_beta_improved,best_alpha_improved)
        tic_6 = timer()
        improved_H_accuracy, improved_H_f1, improved_H_P, improved_H_R = compute_ac(train_data,test_data, "NH", knn_k,granular_balls_original, granular_balls_improved, best_beta_original,best_alpha_original, best_beta_improved,best_alpha_improved)
        tic_7 = timer()
        ML_knn_accuracy, ML_knn_f1, ML_knn_P, ML_knn_R = compute_mlknn_ac(data_ml_knn, target_ml_knn, train_data_index,
                                                                          test_data_index,knn_k)
        tic_8 = timer()

        nc_accuracy, nc_f1, nc_P, nc_R = compute_nc_ac(train_data, test_data)
        tic_9 = timer()

        # ________________________待填坑___________________________________
        # 十折交叉验证的十次准确率
        traditional_accuracy_list.append(traditional_accuracy)
        old_accuracy_list.append(old_accuracy)
        new_accuracy_list.append(new_accuracy)
        original_accuracy_list.append(original_accuracy)
        improved_accuracy_list.append(improved_accuracy)
        original_H_accuracy_list.append(original_H_accuracy)
        improved_H_accuracy_list.append(improved_H_accuracy)
        ml_knn_accuracy_list.append(ML_knn_accuracy)
        nc_accuracy_list.append(nc_accuracy)

        # 十折交叉验证的十次F1评分
        traditional_f1_list.append(traditional_f1)
        old_f1_list.append(old_f1)
        new_f1_list.append(new_f1)
        original_f1_list.append(original_f1)
        improved_f1_list.append(improved_f1)
        original_H_f1_list.append(original_H_f1)
        improved_H_f1_list.append(improved_H_f1)
        ml_knn_f1_list.append(ML_knn_f1)
        nc_f1_list.append(nc_f1)

        traditional_P_list.append(traditional_P)
        old_P_list.append(old_P)
        new_P_list.append(new_P)
        original_P_list.append(original_P)
        improved_P_list.append(improved_P)
        original_H_P_list.append(original_H_P)
        improved_H_P_list.append(improved_H_P)
        ml_knn_P_list.append(ML_knn_P)
        nc_P_list.append(nc_P)

        traditional_R_list.append(traditional_R)
        old_R_list.append(old_R)
        new_R_list.append(new_R)
        original_R_list.append(original_R)
        improved_R_list.append(improved_R)
        original_H_R_list.append(original_H_R)
        improved_H_R_list.append(improved_H_R)
        ml_knn_R_list.append(ML_knn_R)
        nc_R_list.append(nc_R)

        # 十折交叉验证的十次执行时间
        traditional_time = tic_1 - tic_0
        old_time = tic_2 - tic_1
        new_time=tic_3 - tic_2
        original_time = tic_4 - tic_3
        improved_time = tic_5 - tic_4
        original_H_time = tic_6 - tic_5
        improved_H_time = tic_7 - tic_6
        ml_knn_time = tic_8 - tic_7
        nc_time = tic_9 - tic_8
        #
        traditional_time_list.append(traditional_time)
        old_time_list.append(old_time)
        new_time_list.append(new_time)
        original_time_list.append(original_time)
        improved_time_list.append(improved_time)
        original_H_time_list.append(original_H_time)
        improved_H_time_list.append(improved_H_time)
        ml_knn_time_list.append(ml_knn_time)
        nc_time_list.append(nc_time)


    # 不同数据集的准确率均值
    KNN_accuracy.append(np.mean(traditional_accuracy_list))
    GB_accuracy_k.append(np.mean(old_accuracy_list))
    DBGB_accuracy_k.append(np.mean(new_accuracy_list))
    GB_accuracy.append(np.mean(original_accuracy_list))
    DBGB_accuracy.append(np.mean(improved_accuracy_list))
    GB_H_accuracy.append(np.mean(original_H_accuracy_list))
    DBGB_H_accuracy.append(np.mean(improved_H_accuracy_list))
    ml_knn_accuracy.append(np.mean(ml_knn_accuracy_list))
    n_c_accuracy.append(np.mean(nc_accuracy_list))

    print("accuracy:",GB_accuracy_k,DBGB_accuracy_k,GB_accuracy, DBGB_accuracy,GB_H_accuracy,DBGB_H_accuracy,KNN_accuracy,
          ml_knn_accuracy,n_c_accuracy)

    # 不同数据集的f1_score均值
    KNN_f1.append(np.mean(traditional_f1_list))
    GB_f1_k.append(np.mean(old_f1_list))
    DBGB_f1_k.append(np.mean(new_f1_list))
    GB_f1.append(np.mean(original_f1_list))
    DBGB_f1.append(np.mean(improved_f1_list))
    GB_H_f1.append(np.mean(original_H_f1_list))
    DBGB_H_f1.append(np.mean(improved_H_f1_list))
    ml_knn_f1.append(np.mean(ml_knn_f1_list))
    n_c_f1.append(np.mean(nc_f1_list))
    print("f1:",GB_f1_k, DBGB_f1_k,GB_f1, DBGB_f1,GB_H_f1,DBGB_H_f1,KNN_f1,ml_knn_f1,n_c_f1)

    KNN_P.append(np.mean(traditional_P_list))
    GB_P_k.append(np.mean(old_P_list))
    DBGB_P_k.append(np.mean(new_P_list))
    GB_P.append(np.mean(original_P_list))
    DBGB_P.append(np.mean(improved_P_list))
    GB_H_P.append(np.mean(original_H_P_list))
    DBGB_H_P.append(np.mean(improved_H_P_list))
    ml_knn_P.append(np.mean(ml_knn_P_list))
    n_c_P.append(np.mean(nc_P_list))
    print("P:",GB_P_k,DBGB_P_k,GB_P, DBGB_P,GB_H_P, DBGB_H_P,KNN_P,ml_knn_P,n_c_P)

    KNN_R.append(np.mean(traditional_R_list))
    GB_R_k.append(np.mean(old_R_list))
    DBGB_R_k.append(np.mean(new_R_list))
    GB_R.append(np.mean(original_R_list))
    DBGB_R.append(np.mean(improved_R_list))
    GB_H_R.append(np.mean(original_H_R_list))
    DBGB_H_R.append(np.mean(improved_H_R_list))
    ml_knn_R.append(np.mean(ml_knn_R_list))
    n_c_R.append(np.mean(nc_R_list))
    print("R:",GB_R_k,DBGB_R_k, GB_R, DBGB_R,GB_H_R,DBGB_H_R, KNN_R,ml_knn_R,n_c_R)



    # 不同数据集的执行时间均值
    KNN_runtime.append(np.mean(traditional_time_list))
    GB_runtime_k.append(np.mean(old_time_list))
    DBGB_runtime_k.append(np.mean(new_time_list))
    GB_runtime.append(np.mean(original_time_list))
    DBGB_runtime.append(np.mean(improved_time_list))
    GB_H_runtime.append(np.mean(original_H_time_list))
    DBGB_H_runtime.append(np.mean(improved_H_time_list))
    ml_knn_runtime.append(np.mean(ml_knn_time_list))
    n_c_runtime.append(np.mean(nc_time_list))
    print("runtime:", GB_runtime_k, DBGB_runtime_k,GB_runtime, DBGB_runtime,GB_H_runtime,DBGB_H_runtime, KNN_runtime,ml_knn_runtime,n_c_runtime)

    data = {'KNN':[KNN_accuracy,KNN_f1,KNN_R,KNN_P,KNN_runtime],
            'ML-KNN':[ml_knn_accuracy, ml_knn_f1, ml_knn_R, ml_knn_P, ml_knn_runtime, ],
            'NC': [n_c_accuracy, n_c_f1, n_c_R, n_c_P, n_c_runtime, ],
            'GBKNN': [GB_accuracy_k, GB_f1_k, GB_R_k, GB_P_k, GB_runtime_k, ],
            'GBKNN-k': [GB_H_accuracy, GB_H_f1, GB_H_R, GB_H_P, GB_H_runtime, ],
            'DBGBKNN-1' : [DBGB_accuracy_k, DBGB_f1_k, DBGB_R_k, DBGB_P_k, DBGB_runtime_k, ],
            'DBGBKNN-K': [DBGB_H_accuracy, DBGB_H_f1, DBGB_H_R, DBGB_H_P, DBGB_H_runtime, ],
            'GBKNN-3WD': [GB_accuracy, GB_f1, GB_R, GB_P, GB_runtime, ],
            'DBGBKNN-3WD': [DBGB_accuracy, DBGB_f1, DBGB_R, DBGB_P, DBGB_runtime, ],
            }
    index_names = ['Accuracy', 'F1', 'Re', 'P', 'Runtime']

    #  使用pandas创建DataFrame，并指定行名
    df = pd.DataFrame(data, index=index_names)
    # csv_name = 'test' +'-'+ filename + '.csv'
    # 输出表格到控制台
    print(df)
    # print(max(improved_accuracy_list),min(improved_accuracy_list))
    # # 输出DataFrame为CSV文件
    df.to_csv('result/variousKNN-3WD/粒球-KNN-3WD-test-2023-09-03.csv', mode='a', index_label=filename, index=True)

