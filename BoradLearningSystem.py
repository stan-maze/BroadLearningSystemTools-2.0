import numpy as np
import pandas as pd

from scipy import linalg as LA
from numpy import random
from sklearn import preprocessing



class BLS:
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30, is_argmax=True):
        self.FeatureNodes = NumFeatureNodes
        self.FeatureWindows = NumWindows  
        self.EnhancementNodes = NumEnhance
        self.S = S
        self.C = C
        self.is_argmax = is_argmax

    def GridSearchCV_BLS(NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30):
        return 0

    # tansig也是常用的激活函数, 双曲正切, sigmoid也称logsig
    def tansig(self, x):
        return (2/(1+np.exp(-2*x)))-1
    
    def relu(self, x):
        return np.maximum(0, x)

    def pinv(self, A, reg):
        # 公式2, reg是λ, 注意是At*A, 论文确实有误
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def cls_pinv(self, matrix):
        # 分类情况, 只改用极小的C? 对应公式3
        return np.mat(self.C * np.eye(matrix.shape[1]) + matrix.T.dot(matrix)).I.dot(matrix.T)

    '''
    参数压缩
    '''
    # 还不能理解, 软阈值操作
    # 输出的z是经过软阈值处理后的结果。如果输入矩阵a的某个元素大于b，那么相应的z元素将等于a的该元素减去b；如果a的某个元素小于-b，那么相应的z元素将等于a的该元素加上b；如果在-b和b之间，相应的z元素将等于0。
    # 这种软阈值操作在信号处理、统计学和机器学习中常用于稀疏性推动，使得一些元素趋向于零，从而达到降维或者特征选择的效果。
    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    '''
    参数稀疏化
    '''
    # 基于ADMM的Sparse Autoencoder
    # A对应Z,b对应X, 在给定的变换结果Z和原数据X上, 找到一个更稀疏的变换矩阵W
    def sparse_bls(self, A, b):
        # A一组特征[数据量,FeatureNodes], b原数据[数据量,特征维数(+1)]
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk


class BLSRegressor(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30):
        super().__init__()

    def fit(self, train_x, train_y):
        u = 0
        WF = list()
        # 总共有FeatureNodes个特征节点,每个节点有FeatureWindows大小, 不知道为什么不按照FeatureNodes来迭代, 可能是特征选择的通常做法?
        for i in range(self.FeatureWindows):
            random.seed(i+u)
            WeightFea = 2*random.randn(train_x.shape[1]+1, self.FeatureNodes)-1
            WF.append(WeightFea)
        # 增强节点的第一种形式 nk->h, 有EnhancementNodes个， 为什么nk+1?
        self.WeightEnhan = 2*random.randn(self.FeatureWindows*self.FeatureNodes+1, self.EnhancementNodes)-1

        # 出于还没搞明白的原因，数据拼接了一个0.1，上面的权重矩阵也对应是train_x.shape[1]+1, 人为添加bias吗
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        
        y = np.zeros([train_x.shape[0], self.FeatureWindows*self.FeatureNodes])
        self.WFSparse = list()
        self.distOfMaxAndMin = np.zeros(self.FeatureWindows)
        self.meanOfEachWindow = np.zeros(self.FeatureWindows)
        for i in range(self.FeatureWindows):
            WeightFea = WF[i]
            # A1那是[i, n]中的一组值, 换句话说就是 特征节点的第i个值的组合
            A1 = H1.dot(WeightFea)
            # 出于还没搞明白的原因，这里scaler了一下
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            # A一组特征[数据量,FeatureNodes], b原数据[数据量,特征维数(+1)]
            WeightFeaSparse = self.sparse_bls(A1, H1).T
            self.WFSparse.append(WeightFeaSparse)
            # 保持变换结果A1的更稀疏的权重, T1是在该权重下A1的近似
            T1 = H1.dot(WeightFeaSparse)
            # 将T1中心化
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            # 将变换结果T1记入y, 按照一组一组, 第i个FeatureWindows里面是所有FeatureNodes对应值的排列
            y[:, self.FeatureNodes * i:self.FeatureNodes * (i+1)] = T1
        
        # 特征节点输入之前也要拼接0.1, 对应项目权重矩阵的nk+1
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        # 同样直接使用初始化的WeightEnhan得到增强节点, 这里没有再进一步稀疏化
        T2 = H2.dot(self.WeightEnhan)
        T2 = self.tansig(T2)
        # 最终的输入,将Z(y)和H拼接
        T3 = np.hstack([y, T2])
        # 文章公式2, 就是计算伪逆然后得到W, 不还是用self.c??
        self.WeightTop = self.pinv(T3, self.C).dot(train_y)

        print(self.WeightTop.shape)
        NetoutTrain = T3.dot(self.WeightTop.T)

        return NetoutTrain

    def predict(self, test_x):
        # 使用训练得到的权重, 按顺序把所有节点值计算好
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.FeatureWindows * self.FeatureNodes])
        for i in range(self.FeatureWindows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            # 注意仍然要中心化
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.FeatureNodes * i:self.FeatureNodes*(i+1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop.T)

        return NetoutTest


class BLSClassifier(BLS):
    def __init__(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30):
        super().__init__()

    def fit(self, train_x, train_y, is_excel_label=False):
        """模型本体"""

        if is_excel_label:
            train_y = [[i] for i in train_y]
            encoder = preprocessing.OneHotEncoder()
            encoder.fit(train_y)
            train_y = encoder.transform(train_y).toarray()

        # --Train--
        train_x = preprocessing.scale(train_x, axis=1)  # 标准化处理样本
        Feature_InputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 将输入矩阵进行行链接，即平铺展开整个矩阵
        Output_FeatureMappingLayer = np.zeros([train_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        self.Beta1_EachWindow = []
        self.Dist_MaxAndMin = []
        self.Min_EachWindow = []
        self.ymin = 0
        self.ymax = 1

        # 特征层
        for i in range(self.FeatureWindows):
            random.seed(i + 2022)
            W_EachWindow = 2 * random.randn(train_x.shape[1] + 1, self.FeatureNodes) - 1  # 随机化特征层初始权重
            Feature_EachWindow = np.dot(Feature_InputDataWithBias, W_EachWindow)  # 计算每个特征映射中间态

            # scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Feature_EachWindow)                      # 对上述结果归一化处理
            # Feature_EachWindowAfterPreprocess = scaler1.transform(Feature_EachWindow)                               # 进行标准化

            Feature_EachWindowAfterPreprocess = Feature_EachWindow  # 进行标准化
            Beta_EachWindow = self.sparse_bls(Feature_EachWindowAfterPreprocess, Feature_InputDataWithBias).T  # 随机化特征映射初始偏置
            self.Beta1_EachWindow.append(Beta_EachWindow)
            Output_EachWindow = np.dot(Feature_InputDataWithBias, Beta_EachWindow)  # 计算每个特征映射最终输出

            self.Dist_MaxAndMin.append(np.max(Output_EachWindow, axis=0) - np.min(Output_EachWindow, axis=0))  # 计算损失函数
            self.Min_EachWindow.append(np.min(Output_EachWindow, axis=0))
            # 这里用的是最小值来中心化? 为什么称之为计算损失函数?
            Output_EachWindow = (Output_EachWindow - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i]
            # 计算特征层最终输出
            Output_FeatureMappingLayer[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = Output_EachWindow

        # 这里使用了什么技巧?为什么特征节点和增强节点分开计算了?
        train_ori_hance = np.hstack([train_x])
        train_x_enhance = preprocessing.scale(train_ori_hance, axis=1)
        Input_EnhanceLayerWithBias = np.hstack([train_x_enhance, 0.1 * np.ones((train_x_enhance.shape[0], 1))])
        
        # 为什么使用LA.orth嵌套了一下增强节点的权重矩阵?
        # 使用 SVD 为 A 的范围构造正交基, 希望W的权重是正交的,本身增强节点是应该使用FeatureMapping的
        # 使用正交的可以使增强节点的区分度也加大吧
        self.W_EnhanceLayer = LA.orth(2 * random.randn(train_x_enhance.shape[1] + 1, self.EnhancementNodes)) - 1

        # 为什么用self.S?
        Temp_Output_EnhanceLayer = np.dot(Input_EnhanceLayerWithBias, self.W_EnhanceLayer)  # 计算增强层中间态
        self.Parameter_Shrink = self.S / np.max(Temp_Output_EnhanceLayer)
        Output_EnhanceLayer = self.relu(Temp_Output_EnhanceLayer * self.Parameter_Shrink)  # 计算增强层最终输出

        # 输出层
        Input_OutputLayer = np.hstack([Output_FeatureMappingLayer, Output_EnhanceLayer])  # 合并特征层和增强层作为输出层输入
        Pinv_Output = self.cls_pinv(Input_OutputLayer)  # 计算伪逆

        # 计算系统总权重
        self.W = np.dot(Pinv_Output, train_y)

        OutputOfTrain = np.dot(Input_OutputLayer, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTrain.argmax(axis=1)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTrain
        
        return predlabel
    
    
    def predict(self, test_x):
        test_x = preprocessing.scale(test_x, axis=1)
        Feature_InputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        Output_FeatureMappingLayerTest = np.zeros([test_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        for i in range(self.FeatureWindows):
            Output_EachWindowTest = np.dot(Feature_InputDataWithBiasTest, self.Beta1_EachWindow[i])
            # 这里为什么不对称?self.ymax, self.ymin并没有改动过, 有什么伏笔吗, 还有什么更完善的实现方式?
            Output_FeatureMappingLayerTest[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = (self.ymax - self.ymin) * (Output_EachWindowTest - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i] - self.ymin

        # 这里的增强节点不是由特征节点计算来的, 那跟单层前馈有什么区别
        test_ori_hance = np.hstack([test_x])
        test_x_enhance = test_ori_hance
        Input_EnhanceLayerWithBiasTest = np.hstack([test_x_enhance, 0.1 * np.ones((test_x_enhance.shape[0], 1))])
        Temp_Output_EnhanceLayerTest = np.dot(Input_EnhanceLayerWithBiasTest, self.W_EnhanceLayer)
        Output_EnhanceLayerTest = self.relu(Temp_Output_EnhanceLayerTest * self.Parameter_Shrink)
        Input_OutputLayerTest = np.hstack([Output_FeatureMappingLayerTest, Output_EnhanceLayerTest])  # 合并特征层和增强层作为测试输出层输入

        OutputOfTest = np.dot(Input_OutputLayerTest, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTest.argmax(axis=1)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTest
        
        return predlabel
