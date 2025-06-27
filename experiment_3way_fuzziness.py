#
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold
from pandas import DataFrame
# 数据集读取
from sklearn.preprocessing import MinMaxScaler
from improved_GB import DBGBList

from original_GB import GBList

from tool import compute_best_beta,get_membership_table

def new_compute_x_membership(x, GBList):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    for i in range(len(GBList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        # distance1 = calculate_distance(x, GBList.granular_balls[i].gen_interval_li)
        distance1 = np.linalg.norm(x - GBList.granular_balls[i].center) - \
                    GBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    x_belong_value=0
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = GBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        for i in range(5):
            gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
        x_belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
    return x_belong_value

def get_membership_table_(test_data, granular_balls_improved):
    test_data_no_label = test_data[:, :-1]
    membership_table = []
    for i in range(len(test_data_no_label)):
        x = test_data_no_label[i]
        x_membership = new_compute_x_membership(x, granular_balls_improved)
        membership_table.append(x_membership)
    membership_table_df = DataFrame(membership_table)
    return membership_table_df


def calculate_av_fuzziness_(membership_list):
    fuzziness = 0
    for value in membership_list.values:
        if 0 < value < 1:
            single_fuzziness = 4 * value * (1 - value)
            fuzziness += single_fuzziness
    fuzziness = fuzziness / len(membership_list)
    # print(len(membership_list))
    return float(fuzziness)


def calculate_3way_fuzziness_(membership_list, beta, alpha):
    count = 0
    for value in membership_list.values:
        if beta < value < alpha:
            count += 1
    # if beta == 1 - alpha:
    #     fuzziness = 2 * count * (4 * beta * beta * beta - 6 * beta * beta + 1) / (self.Data_row * 3)
    # else:
    fuzziness = 4 * count * (0.5 * alpha * alpha - (1 / 3) * alpha * alpha * alpha - 0.5 * beta * beta + (
            1 / 3) * beta * beta * beta) / len(membership_list)
    return fuzziness




# 效果好的数据集：'dataR2(116,9)''hcv(596,10)''breast-cancer(699,9)''wine(1143,11)''car(1728,6)''banana(5330,2)'
DBGBKNN_av_fuzziness=[]
DBGBKNN_3wd_fuzziness=[]
_av_fuzziness_list=[]
_3wd_fuzziness_list=[]
# 
# filenames=['hcv(596,10)','breast-cancer(699,9)','Energy Effciency(768,8)','Endgame(958,9)','heart(1025,13)',
#            'car(1728,6)','spambase(4601,57)','banana(5300,2)','elect(10000,13)','Dry_Bean_Dataset(13611,16)',
#              'HTRU(17000,8)','HTRU_2(17898,8)']
# filenames=['HTRU(17898,8)',]
for filename in filenames:
    '''数据读取与预处理'''
    # 效果比较好的数据集
    df = pd.read_csv("datasets/" + filename + ".csv")
    # new文件夹下的数据集
    # df = pd.read_csv("备用数据集/new/" + filename + ".csv")
    # old数据集(被筛选过的备用数据集)
    # df = pd.read_csv("备用数据集/" + filename + ".csv")
    print("———————————————————————————正在处理数据集:" + filename + "—————————————————————————————————————————————")
    # data = df.values
    '''归一化数据集'''
    da1 = df.values
    min_max = MinMaxScaler()
    da2 = min_max.fit_transform(da1[:, :-1])
    print(da2)
    data3 = np.hstack([da2, da1[:, -1:]])
    data = np.unique(data3, axis=0)

    '''十折交叉验证'''
    # 原始十折交叉验证
    # k_fold = StratifiedKFold(n_splits=10)
    # k_fold = k_fold.split(data, data[:, -1])
    # 随机排列交叉验证
    k_fold = ShuffleSplit(n_splits=10)
    k_fold = k_fold.split(data)

    for k, (train_data_index, test_data_index) in enumerate(k_fold):
        # if k == 0:
        print('迭代次数：{}'.format(k + 1))#填充
        print('训练数据长度：{}'.format(len(train_data_index)))
        print('测试数据长度：{}'.format(len(test_data_index)))
        train_data = data[train_data_index, :]
        test_data = data[test_data_index, :]
        # 计算测试数据在密度峰值粒球前提下的隶属度列表
        # 1.生成密度峰值粒球
        granular_balls_original = GBList(train_data)  # create the list of granular balls
        granular_balls_original.init_granular_balls()  # initialize the list
        # 是否删除元素小于2的粒球对结果影响很大
        granular_balls_original.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample
        granular_balls_improved = DBGBList(granular_balls_original)  # create the list of granular balls
        granular_balls_improved.init_granular_balls()  # initialize the list
        granular_balls_improved.del_balls(num_data=2)
        print("granular_balls_improved.granular_balls.num:", len(granular_balls_improved.granular_balls))

        # 2.计算测试数据的隶属度列表
        membership_list = get_membership_table_(test_data,granular_balls_improved)
        av_fuzziness = calculate_av_fuzziness_(membership_list)
        membership_table=get_membership_table(granular_balls_improved)
        beta, alpha, min_fuzziness, = compute_best_beta(granular_balls_improved, membership_table)
        _3wd_fuzziness = calculate_3way_fuzziness_(membership_list, beta, alpha, )
        _av_fuzziness_list.append(av_fuzziness)
        _3wd_fuzziness_list.append(_3wd_fuzziness)
        print(_av_fuzziness_list)
        print(_3wd_fuzziness_list)
    DBGBKNN_av_fuzziness.append(np.mean(_av_fuzziness_list))
    DBGBKNN_3wd_fuzziness.append(np.mean( _3wd_fuzziness_list))
    print("1",DBGBKNN_av_fuzziness)
    print('1',DBGBKNN_3wd_fuzziness)

# 将不同数据集不同算法下的准确率画为图形并保存
# labels = ['HCV','Breast-cancer','Car','Banana','Heart','HTRU_2','Dry_Bean','Endgame','HTRU','Elect','mushroom','Spambase',]
labels = [r'$\mathit{D}_1$',r'$\mathit{D}_2$',r'$\mathit{D}_3$',r'$\mathit{D}_4$',r'$\mathit{D}_5$',r'$\mathit{D}_6$',
        r'$\mathit{D}_7$',r'$\mathit{D}_8$',r'$\mathit{D}_9$',r'$\mathit{D}_{10}$',r'$\mathit{D}_{11}$',r'$\mathit{D}_{12}$',]
#'b','c','d','e','f','g','h','i','j','k','l',
#Banknote','Concrete','Contraceptive','Energy Efciency']
x = np.arange(len(labels))
width = 0.2
plt.figure(figsize=(10, 6))
plt.bar(x -0.5*width, DBGBKNN_av_fuzziness, width, label='GBNRS++')
plt.bar(x +0.5*width, DBGBKNN_3wd_fuzziness, width, label='3WC-GBNRS++')
plt.xlabel('The various datasets')
plt.ylabel('The fuzziness loss of various decisions')
plt.xticks(x, labels)
plt.legend(frameon=False, loc='upper left', )
plt.savefig('result/2D', dpi=600, bbox_inches='tight')
plt.show()
