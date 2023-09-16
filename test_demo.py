from BoradLearningSystem import BLSRegressor, BLSClassifier
from BroadLearningSystemBasedAutoEncoder import BLSAEExtractor
from sklearn.datasets import load_iris, load_breast_cancer

# -- 模型初始化 --
# regressor = BLSRegressor(NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30)
# classifier = BLSClassifier()
extractor = BLSAEExtractor()


# -- mat文件 加载minst --
# dataFile = 'code/mnist.mat'
# data = scio.loadmat(dataFile)
# traindata = np.double(data['train_x']/255)
# trainlabel = np.double(data['train_y'])
# testdata = np.double(data['test_x']/255)
# testlabel = np.double(data['test_y'])


# -- sklearn 加载回归数据 --
# dataset = load_iris()
# data, label = dataset['data'], dataset['target']
# print(label)

# -- sklearn 加载分类数据 --
cls_dataset = load_breast_cancer()
cls_data, cls_label = cls_dataset['data'], cls_dataset['target']
print(cls_data.shape)

# -- BLSAE --
feature_encode_x = extractor.fit(cls_data, cls_label)
print(feature_encode_x.shape)

# -- 回归 --
# train_output = regressor.fit(data, label)
# print(train_output.shape)
# predict_output = regressor.predict(data)
# print(predict_output.shape)

# -- 分类 --
# train_output = classifier.fit(cls_data, cls_label, is_excel_label=True)
# print(train_output)
# predict_output = classifier.predict(cls_data)
# print(predict_output)

