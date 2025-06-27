"""
original granular ball
"""
import numpy as np
from collections import Counter

from numpy import mean
from sklearn.cluster import k_means, DBSCAN
import warnings

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


class GranularBall:
    """class of the granular ball"""

    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.data_no_label = data[:, :-1]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity = self.__get_label_and_purity()
        self.radius = self.__get_gbradis()
        # self.cluster_new_label = self.get_cluster_new_label()
        # self.cluster_num = sum(np.unique(self.cluster_new_label) != -1)

    def __get_gbradis(self):  # 获取粒球的半径
        return mean(np.sqrt(np.sum(np.asarray(self.center - self.data_no_label) ** 2, axis=1)))

    # def get_cluster_new_label(self, eps=0.2, min_samples=2):  # 获取粒球的密度聚类数目
    #     min_max = MinMaxScaler()
    #     gb_data = min_max.fit_transform(self.data_no_label)
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #     labels = dbscan.fit_predict(gb_data)
    #     return labels

    def __get_label_and_purity(self):
        """
        :return: the label and purity of the granular ball.
        """
        count = Counter(self.data[:, -1])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        return label, purity

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        label_cluster = k_means(X=self.data_no_label, n_clusters=2)[1]  # k_means函数返回3个值：k
        # 个聚类中心，每个对象所属划分粒球的标号，各个点到聚类中心的距离平方
        if sum(label_cluster == 0) and sum(label_cluster == 1):  # 划分出的每个粒球中包含至少一个对象
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2

    # def split_kballs(self):
    #     # 计算k个新的聚类中心
    #     new_centers = []
    #     for i in range(self.cluster_num):
    #         new_centers.append(self.data_no_label[self.cluster_new_label == i].mean(0))
    #         new_center_array = np.array(new_centers)
    #     label_cluster = k_means(X=self.data_no_label,
    #                             n_clusters=self.cluster_num, init=new_center_array)[1]
    #     balls = []
    #     for j in range(self.cluster_num):
    #         balls.append(GranularBall(self.data[label_cluster == j, :]))
    #     return balls


class GBList:
    """class of the list of granular ball"""

    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, purity=1.0, min_sample=1):  # 初始化粒球：纯度为1，最小样本数1
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)  # 粒球数量
        i = 0
        while True:
            # 如果当粒球的纯度小于所设置的纯度1，且当前粒球中元素的数量大于最小样本数1
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_balls = self.granular_balls[i].split_2balls()  # 将当前粒球一分为二，并返回两个粒球对象
                self.granular_balls[i] = split_balls[0]  # 将分开后的第一个粒球赋值给当前粒球
                self.granular_balls.append(split_balls[1])  # 粒球列表添加分开后的第二个粒球
                ll += 1  # 粒球数加1
            else:
                i += 1
            if i >= ll:  # 遍历并生成所有符合要求的粒球后，跳出循环
                break
        self.del_balls(num_data=2)
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_balls(self, purity=0, num_data=0):
        """
        Deleting the balls that meets following conditions from the list, updating self.granular_balls and self.data.
        :param purity: delete the balls that purity is large than this value.
        :param num_data: delete the balls that the number of samples is large than this value.
        :return: None
        """
        self.granular_balls = [ball for ball in self.granular_balls if ball.purity >= purity and ball.num >= num_data]
        self.data = self.get_data()

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        label_cluster = k_means(X=self.data[:, :-1], n_clusters=k, init=self.get_center())[1]
        for i in range(k):
            self.granular_balls[i] = GranularBall(self.data[label_cluster == i, :])

    def re_division(self, i):  # 针对属性约简的再次划分粒球
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        label_cluster = k_means(X=self.data[:, attributes], n_clusters=k,
                                init=self.get_center()[:, attributes], max_iter=1)[1]
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division


# # 属性约简部分
# #def get_attribute_reduction(data):
#     """
#     The main function of attribute reduction.
#     :param data: data set
#     :return: reduced attribute set
#     """
#     num, dim = data[:, :-1].shape
#     index = np.array(range(num)).reshape(num, 1)  # column of index
#     data = np.hstack((data, index))  # Add the index column to the last column of the data
#
#     # step 1.
#     granular_balls = GBList(data)  # create the list of granular balls
#     granular_balls.init_granular_balls()  # initialize the list
#     granular_balls.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample
#
#     # step 2.
#     granular_balls.re_k_means()  # Global k-means clustering as fine tuning.
#     granular_balls.del_balls(purity=1)  # delete the ball wh sample
#
#     # step 3.
#     attributes_reduction = list(range(data.shape[1] - 2))
#     for i in range(data.shape[1] - 2):
#         if len(attributes_reduction) <= 1:
#             break
#
#         the_remove_i = attributes_reduction.index(i)
#         attributes_reduction.remove(i)  # remove the ith attribute
#         gb_division = granular_balls.re_division(the_remove_i)  # divide the data with center of granular balls
#         purity = [round(ball.purity, 3) for ball in gb_division]  # get the purity of the divided granular balls
#
#         if sum(purity) == len(purity):  # if the ith attribute can be reduced 删除属性后仍能得到纯度全为1的粒球
#             # Recreate the new list granular balls with attributes after the reduction
#             # step 1.
#             granular_balls = GBList(np.hstack((data[:, attributes_reduction], data[:, -2:])))
#             granular_balls.init_granular_balls()
#             granular_balls.del_balls(num_data=2)
#
#             # step 2.
#             granular_balls.re_k_means()
#             granular_balls.del_balls(purity=1)
#
#         else:  # If the ith attribute is can't be reduced, then add it back.
#             attributes_reduction.append(i)
#             attributes_reduction.sort()
#     attributes_reduction = attributes_reduction
#  return attributes_reduction
