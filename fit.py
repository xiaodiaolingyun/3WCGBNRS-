
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


def cluster_centers(gamma, LocalR, GlobalR, l):
    N = len(gamma)
    centers = []
    # Calculate differences between adjacent elements in gamma
    d_gamma_s = np.abs(np.diff(gamma))

    for i in range(N - l, -1, -1):
        # print("———————————————————————————正在处理数据集:" + str(i) + "—————————————————————————————————————————————")

        x_data = np.arange(i + 1, i + l + 1)
        y_data = gamma[i:i + l]

        # Use curve_fit to fit a linear model

        popt, _ = curve_fit(lambda x, a, b: a * x + b, x_data, y_data)
        a_i, b_i = popt

        # Calculate predicted gamma_i
        predicted_gamma_i = a_i * (i - 1) + b_i
        # print('1:', i - 1, '2:', predicted_gamma_i)

        delta_gamma_i = abs(gamma[i - 1] - predicted_gamma_i)

        centers.append(i)

        if delta_gamma_i > LocalR * d_gamma_s[i] and delta_gamma_i > GlobalR * np.max(d_gamma_s):
            break

    return len(centers), gamma[:i]




# # # 示例数据
# gamma = np.array([0.21088867, 0.21369751, 0.22020966, 0.22132692, 0.23204089, 0.27172684, 0.27503289, 0.2788295, 0.29990106,
#      0.30829541, 0.32236211, 0.34741149, 0.35039196, 0.35506576, 0.36424776, 0.37207788, 0.37608305, 0.38779762,
#      0.39031878, 0.39377552, 0.41347777, 0.41710921, 0.42397848, 0.42992844, 0.42995701, 0.43608769, 0.44103213,
#      0.44739662, 0.45501409, 0.45776119, 0.45938183, 0.49330931, 0.49971622, 0.52467026, 0.52972465, 0.5316058,
#      0.53852241, 0.54308795, 0.54662548, 0.54906538, 0.5588632, 0.56336278, 0.59310337, 0.60214118, 0.60474048,
#      0.60575045, 0.64750784, 0.65141381, 0.68478663, 0.70672220, 0.70743680, 0.80982723, 0.81365958, 0.85968104,
#      0.88068615])
# LocalR = 4.8
# GlobalR = 0.02
# l=20
#
# num_centers, center_indices = cluster_centers(gamma, LocalR, GlobalR, l)
# print("Number of cluster centers:", num_centers)
# print("Center indices:", center_indices)
#
# # 绘制拟合曲线与真实曲线
# plt.plot(center_indices, gamma[center_indices], 'ro', label="Cluster Centers")
# plt.plot(np.arange(len(gamma)), gamma, label="True Data")
# plt.legend()
# plt.xlabel("Index")
# plt.ylabel("Gamma Value")
# plt.show()




# gamma=np.array([0.07982873,0.08180002 ,0.09327065,0.10764289,0.12395292 ,0.13716012 ,0.13789834,0.14160186 ,0.14195592 ,0.15064079 ,0.15351666 ,
#   0.16653613,0.16697523,0.17811199,0.19233556,0.20524711,0.22461928,0.2261372,0.23085883,0.23435003,0.24541121,0.25202721,0.25402494,0.26112297,
#   0.2705711, 0.27705186, 0.27884316 , 0.28819057, 0.30942748,0.33291292, 0.3542318 , 0.36398818 , 0.38176817,0.39716217,0.39848842,0.44279548 ,
#   0.44388951,0.44958709 ,0.45592537,0.46023611,0.47688405,0.4785782 ,0.48625949,0.49851539,0.50037341,0.52104463,0.55962757,0.59106776,
#   0.61632605,0.67000971])
































































































































































































































