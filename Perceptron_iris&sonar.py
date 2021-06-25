from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron  # 单层感知器模型库函数
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier  # 多层感知机（全连接神经网络）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义函数decision_display，绘制训练后感知器模型给出的决策区域
def decision_display(X, y, classifier):
    # 设定二分类的对应类别数据点的表示形状以及颜色
    markers = ('.', 'o')
    colors = ('blue', 'yellow')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 用pca降维后二维原始数据data绘制网格图
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 0].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    # 对训练集+测试集数据全部做预测的预测分类结果
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    # 绘制等高线（可看作线性可分二分类数据的分界线&非线性分界区间）
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())  # x轴显示范围
    plt.ylim(xx2.min(), xx2.max())  # y轴显示范围

    # 绘制全部数据点的散点图
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label=cl)
    plt.legend()  # 显示图例
    plt.show()


iris = datasets.load_iris()  # 使用iris数据集
pca = decomposition.PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)  # PCA将4维特征维度降维至2维
iris_label = (iris.target != 0) * 1  # 数据标签
# 这里由于已知setosa和其他两类（vericolor和virginca）是线性可分的，但后面两类是线性不可分的
# 而且目的是做二分类问题，故最终决定将数据处理为setosa为一类（标签0），其他两类为另一类（标签1）来做二分类问题
X_train, X_test, y_train, y_test = train_test_split(iris_pca, iris_label, test_size=0.2, random_state=0)

# 训练感知器模型
# max_iter是最大迭代次数，eta0是学习率，
# random_state参数在每次迭代后初始化重新排练数据集
model = Perceptron(max_iter=50, eta0=0.01, random_state=0)
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)  # 用训练好模型给出测试集的预测标签

# 给出测试集分类准确的个数以及测试集数据总个数，并计算感知器在测试数据集上的分类准确率
print('单层感知机线性可分iris二分类预测正确的测试集数据个数/测试集数据总个数:{}/{}'.format(
    (y_test == y_prediction).sum(), y_test.shape[0]))
print('单层感知机线性可分iris二分类准确率:%.2f' % accuracy_score(y_test, y_prediction))
# 分类可视化
decision_display(X=iris_pca, y=iris_label, classifier=model)


# 数据预处理
sonar = pd.read_csv('sonar.all-data', header=None, sep=',')
sonar1 = sonar.iloc[0:208, 0:60]
sonar_data = np.mat(sonar1)  # 转换为矩阵
pca = decomposition.PCA(n_components=2)
sonar_pca = pca.fit_transform(sonar_data)  # PCA将4维特征维度降维至2维
label0 = np.zeros(97)
label1 = np.ones(111)
sonar_label = np.append(label0, label1)  # 设置标签，前97个为rock（label=0），其他为metal（label=1）
X_train, X_test, y_train, y_test = train_test_split(sonar_pca, sonar_label, test_size=0.2, random_state=0)

# 训练感知器模型
# max_iter是最大迭代次数，eta0是学习率，
# random_state参数在每次迭代后初始化重新排练数据集
model = Perceptron(max_iter=50, eta0=0.01, random_state=0)
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)  # 用训练好模型给出测试集的预测标签

# 给出测试集分类准确的个数以及测试集数据总个数，并计算感知器在测试数据集上的分类准确率
print('单层感知机线性不可分sonar二分类预测正确的测试集数据个数/测试集数据总个数:{}/{}'.format(
    (y_test == y_prediction).sum(), y_test.shape[0]))
print('单层感知机线性不可分sonar二分类准确率:%.2f' % accuracy_score(y_test, y_prediction))
# 分类可视化
decision_display(X=sonar_pca, y=sonar_label, classifier=model)

# 改用多层感知机（全连接神经网络）进行线性不可分sonar数据分类
mlp = MLPClassifier(hidden_layer_sizes=(100, 30, 10), max_iter=100000)
# 使用降维后二维数据进行二分类尝试
mlp.fit(X_train, y_train)
y_prediction = mlp.predict(X_test)  # 用训练好模型给出测试集的预测标签

# 给出测试集分类准确的个数以及测试集数据总个数，并计算感知器在测试数据集上的分类准确率
print('降维数据多层感知机线性不可分sonar二分类预测正确的测试集数据个数/测试集数据总个数:{}/{}'.format(
    (y_test == y_prediction).sum(), y_test.shape[0]))
print('降维数据多层感知机线性不可分sonar二分类准确率:%.2f' % accuracy_score(y_test, y_prediction))
# 分类可视化
decision_display(X=sonar_pca, y=sonar_label, classifier=mlp)

# 使用未降维60维数据进行二分类尝试
X_train, X_test, y_train, y_test = train_test_split(sonar_data, sonar_label,
                                                    test_size=0.2,
                                                    random_state=0)
mlp.fit(X_train, y_train)
y_prediction = mlp.predict(X_test)  # 用训练好模型给出测试集的预测标签
# 给出测试集分类准确的个数以及测试集数据总个数，并计算感知器在测试数据集上的分类准确率
print('未降维多层感知机线性不可分sonar二分类预测正确的测试集数据个数/测试集数据总个数:{}/{}'.format(
    (y_test == y_prediction).sum(), y_test.shape[0]))
print('未降维多层感知机线性不可分sonar二分类准确率:%.2f' % accuracy_score(y_test, y_prediction))
# 分类可视化(60维画不出人能理解的图，故不需要可视化)
