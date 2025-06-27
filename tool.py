import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from improved_GB import DBGBList
from original_GB import GBList
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import mean, linspace
import os

def distance(x, y):
    return np.linalg.norm(x - y)

def getDistanceMatrix(gblist_no_label):
    # 计算距离矩阵
    N,D = gblist_no_label.shape
    dists = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            dists[i, j] = distance(gblist_no_label[i,:],gblist_no_label[j,:])
    return dists


def select_dc(dists):
    #方法一：取前百分之二作为dc
    # 计算dc，也可以手动输入，此处没有实现手动输入
    # 不使用np.percentile()，是因为dists有部分重复值、0值
    N = np.shape(dists)[0]
    dists_flat = np.reshape(dists, N * N)
    dists_sort = np.sort(dists_flat)
    position = int(N * (N - 1) * 0.02)
    # 将距离矩阵排序后取第2%个，可以调整该百分比使聚类效果更加
    dc = dists_sort[position + N]
    # # 方法二：用二分法求dc
    # N = np.shape(dists)[0]
    # max_dis = np.max(dists)
    # min_dis = np.min(dists)
    # dc = (max_dis + min_dis) / 2
    # while True:
    #     n_neighs = np.where(dists < dc)[0].shape[0] - N
    #     rate = n_neighs / (N * (N - 1))
    #     if 0.01 <= rate <= 0.02:
    #         break
    #     if rate < 0.01:
    #         min_dis = dc
    #     else:
    #         max_dis = dc
    #     dc = (max_dis + min_dis) / 2
    #     if max_dis - min_dis < 0.000001:
    #         break
    return dc

# 计算每个粒球中心点的局部密度
def calculate_rho(gblist_i_no_label, dc):
    # c=0
    # s=0
    center_dists_list=[]
    N=np.shape(gblist_i_no_label)[0]
    gblist_center =gblist_i_no_label.mean(0)
    for i in range(N):
        center_dists=distance(gblist_center,  gblist_i_no_label[i, :])
        center_dists_list.append(center_dists)
    for j in range(N):
        ball_rho= np.sum(np.exp(-(center_dists_list[j] / dc) ** 2))
    # for j in range(N):
    #     s=np.exp(-(center_dists_list[j] / dc) ** 2)
    #     c+=s
    #     ball_rho=c
    return ball_rho


def get_membership_table(gblist):
    membership_list=[]
    for i in range(len(gblist.granular_balls)):
        ball_purity = gblist.granular_balls[i].purity
        gblist_no_label = gblist.granular_balls[i].data[:, :-1]
        dists=getDistanceMatrix(gblist_no_label)
        dc=select_dc(dists)
        # print("粒球" + str(i + 1) + "中dc为：\n", dc)
        ball_rho = calculate_rho(gblist.granular_balls[i].data_no_label,dc)#待定
        # print("粒球" + str(i + 1) + "中局部密度为：\n", ball_rho)
        membership_q = np.round(ball_purity*(ball_rho+1)/2,10)
        membership_list.append(membership_q)
        #membership_table_df = DataFrame(membership_q_table)
    print("membership_list_num,membership_list:",np.shape(membership_list),membership_list)
    return membership_list

def calculate_av_fuzziness(gblist,membership_list):
    fuzziness_ = 0
    num_1_all=0
    for i in range(len(gblist.granular_balls)):
        num_1= np.shape(gblist.granular_balls[i].data)[0]
        num_1_all += num_1
        value =membership_list[i]
        if 0 < value <1:
            single_fuzziness = 4 * value * (1 - value)*num_1
            fuzziness_ += single_fuzziness
    av_fuzziness = fuzziness_/num_1_all
    # print('calculate_av_fuzziness:',num_2,av_fuzziness)
    return float(av_fuzziness)

def calculate_3way_fuzziness(gblist,membership_list, beta, alpha):
    num_1_all=0
    count=0
    for i in range(len(gblist.granular_balls)):
        num_1 = np.shape(gblist.granular_balls[i].data)[0]
        num_1_all += num_1
        value = membership_list[i]
        if beta < value < alpha:
            count += num_1
    fuzziness = 4 * count*(0.5 * alpha * alpha - (1 / 3) * alpha * alpha * alpha - 0.5 * beta * beta + (
                1 / 3) * beta * beta * beta) /num_1_all   #count指粒球个数，普通例子里面的样本个数
    return fuzziness

def compute_best_beta(gblist,membership_list):
    av_fuzziness = calculate_av_fuzziness(gblist,membership_list)
    # 设置参数迭代步长
    min_fuzziness = av_fuzziness
    best_beta = 0
    best_alpha = 0.5
    for beta in linspace(0.01, 0.5, 50,endpoint = False):
        for alpha in linspace(0.5, 1, 50,endpoint = False):
            _3way_fuzziness = calculate_3way_fuzziness(gblist,membership_list, beta, alpha)
            minus = abs(_3way_fuzziness - av_fuzziness)
            if minus < min_fuzziness:
                min_fuzziness = minus
                best_beta = beta
                best_alpha = alpha
    print('best_beta,best_alpha,min_fuzziness:', best_beta, best_alpha, min_fuzziness)
    return best_beta, best_alpha, min_fuzziness

def fuzziness_difference(gblist,membership_list,beta):
    av_fuzziness = calculate_av_fuzziness(gblist,membership_list)
    alpha_array = np.round(np.arange(0.5, 1, 0.01), 4)
    beta_list = []
    alpha_list=[]
    fuzziness_difference_list = []
    for j in range(len(alpha_array)):
        alpha = alpha_array[j]
        fuzziness = calculate_3way_fuzziness(gblist,membership_list,beta,alpha)
        fuzziness_difference_= abs(av_fuzziness - fuzziness)
        fuzziness_difference_list.append(fuzziness_difference_)
        beta_list.append(beta)
        alpha_list.append(alpha)
    # print("fuzziness_difference_list:", fuzziness_difference_list, "beta_list:", beta_list, "alpha_list:",alpha_list)
    return fuzziness_difference_list, beta_list, alpha_list


