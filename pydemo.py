import matplotlib.pyplot as plt
import numpy as np

x, y = np.loadtxt('example.txt',  unpack=True)

plt.scatter(x, y,s=1)

# x_np_list = np.arange(-1, 1, 0.01)
# y_list = [0] * len(x_np_list)   # 创建元素相同的列表
# plt.annotate("", xy=(1.01, 0), xycoords='data', xytext=(-1.01, 0), textcoords='data',
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))  # 画x轴
# plt.annotate("", xy=(0, 1.01), xycoords='data', xytext=(0, -1.01), textcoords='data',
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))  # 画y轴
# plt.text(0.96, 0.05, 'x')  # 标x
# plt.text(0.05, 0.96, 'y')  # 标y
# plt.xlim(-1.01, 1.01)
# plt.ylim(-1.01, 1.01)


plt.show()

