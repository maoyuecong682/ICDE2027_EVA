import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder
import numpy as np

class XGBoostModel:
    def __init__(self, num_classes, im_size, channel):
        # 确保输入维度的计算与LSTM时一致
        if isinstance(im_size, (tuple, list)):
            self.input_features = channel * np.prod(im_size)
        else:
            self.input_features = im_size * channel if channel != 1 else im_size

        # 初始化 XGBoost 模型
        self.model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                      max_depth=5, alpha=10, n_estimators=10)

    def fit(self, X_train, y_train):
        # X_train 是训练特征，y_train 是回归目标
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # X_test 是要进行预测的特征数据
        return self.model.predict(X_test)

    def feature_extractor(self, X):
        # 对于 XGBoost，我们直接返回输入特征，因为 XGBoost 本身不需要额外的特征提取
        return X

