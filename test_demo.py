from BoradLearningSystem import BLSRegressor, BLSClassifier
from BroadLearningSystemBasedAutoEncoder import BLSAEExtractor, StackedBLSAEExtractor
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np
from scipy import io as scio

# -- 模型初始化 --
regressor = BLSRegressor(NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30)
classifier = BLSClassifier()
# extractor = BLSAEExtractor()
# stack_extractor = StackedBLSAEExtractor()
# stack_extractor_2 = StackedBLSAEExtractor(is_multi_feature=True)


# # -- mat文件 加载minst --
# dataFile = 'data/mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])


# -- sklearn 加载回归数据 --
dataset = load_iris()
data, label = dataset['data'], dataset['target']
# print(label)

# -- sklearn 加载分类数据 --
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# cls_dataset = load_breast_cancer()
# cls_data, cls_label = cls_dataset['data'], cls_dataset['target']
# print(cls_data.shape)
# 重写代码
# mnist = fetch_openml("mnist_784")
# X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)
# print(mnist.data.shape)

# # -- BLSAE --
# feature_encode_x = extractor.fit(cls_data)
# print(feature_encode_x.shape)

# # -- Stacked BLSAE --
# feature_encode_x = stack_extractor.fit(cls_data)
# print(feature_encode_x.shape)

# feature_encode_x = stack_extractor_2.fit(cls_data)
# print(feature_encode_x)

# # -- 回归 --
# train_output = regressor.fit(data, label)
# print(train_output.shape)
# predict_output = regressor.predict(data)
# print(predict_output.shape)

# -- 分类 --
from sklearn.datasets import load_digits
dataFile = 'data/mnist.mat'
mnist_data = scio.loadmat(dataFile)
# mnist_data = fetch_openml("mnist_784", version=1, cache=True)
keys = mnist_data.keys()
print(keys)
X,y = mnist_data['Z'], mnist_data['y']
y = np.array([l[0] for l in y ])
# X = X/255
print(X[:2], y[:2])
print(X.shape, y.shape)
# mnist_data = load_digits()
# print(mnist_data['data'][:2], mnist_data['target'][:2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score

train_output = classifier.fit(X_train, y_train, is_excel_label=True)
print(train_output[0])
predict_output = classifier.predict(X_test)
# print(len(predict_output), predict_output[0], y_test[0])
acc = accuracy_score(y_test, predict_output)
print(acc)

