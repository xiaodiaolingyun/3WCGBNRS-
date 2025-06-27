import numpy as np

from matplotlib import pyplot as plt
from timeit import default_timer as timer
from improved_GB import DBGBList
from original_GB import GBList
from pandas import DataFrame
import math
from tool import get_membership_table,compute_best_beta
from fit import cluster_centers
# 使用传统KNN算法计算标签
def KNN_compute_label(x, train_data, knn_k):
    # 遍历训练集列表，计算目标点到各个样本点的距离：
    train_data_no_labels = train_data[:, :-1]
    distances = []
    for i in range(len(train_data_no_labels)):
        distance1 = np.linalg.norm(x - train_data_no_labels[i])
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]

    # print('传统计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # print(distances)
    gb_labels = []
    for i in range(knn_k):
        gb_labels.append(train_data[int(distances[i][1]), -1])
    # print(gb_labels)
    # 这里假设为2分类 且标签要么为0 要么为1
    if sum(gb_labels) >= len(gb_labels) / 2:
        x_label = 1
    else:
        x_label = 0
    # print('传统计算标签的时间为：{}'.format(tic_3 - tic_2))
    return x_label

# 使用GBKNN算法计算标签,k=1
def GBNRS_compute_label(x, gbList,):
    # plt.scatter(x[0], x[1], c='black', marker='o', s=60)
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    tic_0 = timer()
    for i in range(len(gbList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - gbList.granular_balls[i].center) - \
                    gbList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # print(distances)
    # 通过距离列表计算K近邻的K值   以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = gbList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        # gb_labels = []
        # for i in range(knn_k):
        #    gb_labels.append(gbList.granular_balls[int(distances[i][1])].label)
        # x_label = max(gb_labels, key=gb_labels.count)
        x_label = gbList.granular_balls[int(distances[0][1])].label
    # print('粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
    # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    return x_label
# 使用GBKNN算法计算标签,k=k*
def GBKNN_compute_label(x, GBList,):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    l = int(len(GBList.granular_balls) * 0.95)
    distances = []
    for i in range(len(GBList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - GBList.granular_balls[i].center) - \
                    GBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = GBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
        for i in range(len(cluster_centers_)):
            for j in range(len(distances[:, 0])):
                if cluster_centers_[i] == distances[j][0]:
                    gb_labels.append(GBList.granular_balls[int(distances[j][1])].label)
        belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        # gb_labels = []
        # for i in range(knn_k):
        #     gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
        # # print(gb_labels)
        # belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
        # maxlabel = gb_labels.count(max(gb_labels, key=gb_labels.count))
        # belong_value = maxlabel/ (len(gb_labels) + float( "1e-8"))
        if belong_value >=0.5:
            x_label = 1
        elif belong_value < 0.5:
            x_label = 0
        # print('密度粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
        # print("计算后取前" + str(k) + "个粒球数量最多的的标签")
        # print("x的标签为：" + str(x_label))
    return x_label

# 使用DBSCAN粒球3WD算法计算标签
def DBGB_3way_compute_label(x, DBGBList, best_beta, best_alpha):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    # l = int(len(DBGBList.granular_balls) * 0.953)
    distances = []
    for i in range(len(DBGBList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - DBGBList.granular_balls[i].center) - \
                    DBGBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = DBGBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        # num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
        # for i in range(len(cluster_centers_)):
        #     for j in range(len(distances[:, 0])):
        #         if cluster_centers_[i] == distances[j][0]:
        #             gb_labels.append(DBGBList.granular_balls[int(distances[j][1])].label)
        # belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        gb_labels = []
        for i in range(5):
            gb_labels.append(DBGBList.granular_balls[int( distances[i][1])].label)
        # print(gb_labels)
        belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
        maxlabel = gb_labels.count(max(gb_labels, key=gb_labels.count))
        belong_value = maxlabel/ (len(gb_labels) + float( "1e-8"))
        if belong_value >best_alpha:
            x_label = 1
        elif belong_value < best_beta:
            x_label = 0
        else:
            x_label = -1
        # print('密度粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
        # print("计算后取前" + str(k) + "个粒球数量最多的的标签")
        # print("x的标签为：" + str(x_label))
    return x_label

# 使用DBSCAN粒球KNN算法计算标签,k=1
def DBGB_compute_label(x, gbList):
    # plt.scatter(x[0], x[1], c='black', marker='o', s=60)
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    tic_0 = timer()
    for i in range(len(gbList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - gbList.granular_balls[i].center) - \
                    gbList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # print(distances)
    # 通过距离列表计算K近邻的K值   以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = gbList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        # gb_labels = []
        # for i in range(knn_k):
        #    gb_labels.append(gbList.granular_balls[int(distances[i][1])].label)
        # x_label = max(gb_labels, key=gb_labels.count)
        x_label = gbList.granular_balls[int(distances[0][1])].label
    # print('粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
    # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    return x_label



#
# 使用DBSCAN粒球KNN算法计算标签,k=k*
def DBGB_HAM_compute_label(x, DBGBList,):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    l = int(len(DBGBList.granular_balls) * 0.98)
    distances = []
    for i in range(len(DBGBList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - DBGBList.granular_balls[i].center) - \
                    DBGBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = DBGBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
        for i in range(len(cluster_centers_)):
            for j in range(len(distances[:, 0])):
                if cluster_centers_[i] == distances[j][0]:
                    gb_labels.append(DBGBList.granular_balls[int(distances[j][1])].label)
        belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        # gb_labels = []
        # for i in range(knn_k):
        #     gb_labels.append(DBGBList.granular_balls[int(distances[i][1])].label)
        # # print(gb_labels)
        # belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
        # maxlabel = gb_labels.count(max(gb_labels, key=gb_labels.count))
        # belong_value = maxlabel/ (len(gb_labels) + float( "1e-8"))
        if belong_value >=0.5:
            x_label = 1
        elif belong_value < 0.5:
            x_label = 0
        # print('密度粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
        # print("计算后取前" + str(k) + "个粒球数量最多的的标签")
        # print("x的标签为：" + str(x_label))
    return x_label


# 使用粒球3WD算法计算标签
def GB_3way_compute_label(x, gbList,best_beta,best_alpha):
    # membership_list = get_membership_table(GBList)
    # best_beta, best_alpha, min_fuzziness = compute_best_beta(GBList, membership_list)
    lower=best_beta
    upper=best_alpha
    l = int(len(gbList.granular_balls) * 0.9)
    # plt.scatter(x[0], x[1], c='black', marker='o', s=60)
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    # tic_0 = timer()
    for i in range(len(gbList.granular_balls)):
        # print(gbList.granular_balls[.i].center, gbList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - gbList.granular_balls[i].center) - \
                   gbList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = gbList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
        for i in range(len(cluster_centers_)):
            for j in range(len(distances[:, 0])):
                if cluster_centers_[i] == distances[j][0]:
                    gb_labels.append(gbList.granular_balls[int(distances[j][1])].label)
        belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        # gb_labels = []
        # for i in range(knn_k):
        #     gb_labels.append(gbList.granular_balls[int(distances[i][1])].label)
        # print(gb_labels)
        # belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
        # # maxlabel = gb_labels.count(max(gb_labels, key=gb_labels.count))
        # # belong_value = maxlabel / (len(gb_labels) + float("1e-8"))
        if belong_value >= upper:
            x_label = 1
        elif belong_value < lower:
            x_label = 0
        else:
            x_label = -1
        # # print('密度粒球KNN计算标签的时间为：{}'.format(tic_3 - tic_2))
        # # print("计算后取前" + str(k) + "个粒球数量最多的的标签")
        # # print("x的标签为：" + str(x_label))
    return x_label



# 使用密度粒球KNN算法计算单个元素的隶属度
def new_compute_x_membership(x, GBList):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    tic_0 = timer()
    l = int(len(GBList.granular_balls) * 0.96)
    for i in range(int(len(GBList.granular_balls))):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - GBList.granular_balls[i].center) - \
                    GBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    print('distances:',distances)
    # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    k = 1
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_belong_value = GBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
        for i in range(len(cluster_centers_)):
            for j in range(len(distances[:, 0])):
                if cluster_centers_[i] == distances[j][0]:
                    # print(distances[j][1])
                    gb_labels.append(GBList.granular_balls[int(distances[j][1])].label)
        x_belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        # gb_labels = []
        # for i in range(15):
        #     gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
        # x_belong_value= sum(gb_labels) / (len(gb_labels) + float(1e-8))
        # # gb_labels = []
        # num_centers, cluster_centers_= cluster_centers(distances[:, 0], 0.48, 0.02, 50)
        # for i in range(len(cluster_centers_)):
        #     gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
        # # print(gb_labels)
        # # 这里假设为2分类 且标签要么为0 要么为1
        # x_belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
    return x_belong_value


def compute_ac(train_data,test_data, compute_type,knn_k, granular_balls_original, granular_balls_improved,  best_beta_improved, best_alpha_improved,best_beta_original, best_alpha_original,):
    test_data_no_label = test_data[:, :-1]
    right_num = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    x_label = 0
    for i in range(len(test_data_no_label)):
        x = test_data_no_label[i]
        if compute_type == 'T':
            x_label = KNN_compute_label(x, train_data, knn_k)#knn
        if compute_type == 'OK':
            x_label = GBNRS_compute_label(x, granular_balls_original,)#gbnrs
        elif compute_type == 'NK':
            x_label = DBGB_compute_label(x, granular_balls_improved,)#dbgbnrs
        elif compute_type == 'O3':
            x_label = GB_3way_compute_label(x, granular_balls_original, best_beta_original, best_alpha_original)
        elif compute_type == 'N3':
            x_label = DBGB_3way_compute_label(x, granular_balls_improved, best_beta_improved, best_alpha_improved)
        elif compute_type == 'OH':
            x_label = GBKNN_compute_label(x, granular_balls_original, )#gbknn
        elif compute_type == 'NH':
            x_label = DBGB_HAM_compute_label(x, granular_balls_improved, )#dbgbknn
        else:
            print('计算类型输入错误，无法计算')
        if x_label == test_data[i, -1]:
            right_num += 1
        if test_data[i, -1] == 1 and x_label == 1:
            tp += 1
        elif test_data[i, -1] == 1 and x_label == 0:
            fp += 1
        elif test_data[i, -1] == 0 and x_label == 1:
            fn += 1
        elif test_data[i, -1] == 0 and x_label == 0:
            tn += 1
    accuracy = right_num / (tp+fp+fn+tn+float("1e-8"))
    # 查准率，精确率 越高越好
    P = tp / (tp + fp + float("1e-8"))
    # 查全率，召回率 越高越好
    R = tp / (tp + fn + float("1e-8"))
    # f1评分，数值越大越稳定，（但还要考虑模型的泛化能力，不能造成过拟合）
    f1_score = (2 * P * R) / (P + R + float("1e-8"))
    print("right_num,tp,fp ,fn,tn:",right_num,tp,fp ,fn,tn)
    return accuracy, f1_score, P, R,

