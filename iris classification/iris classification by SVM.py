from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


# 定义字典，将字符与数字对应起来
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


path = 'Iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
# 确定输入和输出
x, y = np.split(data, (4,), axis=1)  # 将data按前4列返回给x作为输入，最后1列给y作为标签值
x = x[:, 0:2]  # 取x的前2列作为svm的输入，为了方便展示

train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
model.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

train_score = model.score(train_data, train_label)
print("训练集：", train_score)
test_score = model.score(test_data, test_label)
print("测试集：", test_score)

# 训练集和测试集的预测结果
trainPredict = (model.predict(train_data).reshape(-1, 1))
testPredict = model.predict(test_data).reshape(-1, 1)

# 将预测结果进行展示,首先画出预测点，再画出分类界面

# 画图例和点集
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 设置x轴范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 设置y轴范围
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])  # 设置点集颜色格式
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 设置边界颜色
plt.xlabel('花萼长度', fontsize=13)  # x轴标注
plt.ylabel('花萼宽度', fontsize=13)  # y轴标注
plt.xlim(x1_min, x1_max)  # x轴范围
plt.ylim(x2_min, x2_max)  # y轴范围
plt.title('鸢尾花SVM二特征分类')  # 标题

# 画出测试点
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)

# 画出预测点，并将预测点圈出
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2, cmap=cm_dark)
# 画分类界面；生成网络采样点
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
# 测试点
grid_test = np.stack((x1.flat, x2.flat), axis=1)
# 预测分类值
grid_hat = model.predict(grid_test)
# 使之与输入的形状相同
grid_hat = grid_hat.reshape(x1.shape)
# 预测值显示
plt.pcolormesh(x1, x2, grid_hat, shading='auto', cmap=cm_light)
plt.show()
