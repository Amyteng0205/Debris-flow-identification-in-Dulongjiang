# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:48:03 2023

@author: pc
"""

from joblib import load
import pandas as pd

#读取数据
df = pd.read_excel('H:/test/新数据集/数据集11.xlsx',sheet_name='数据集11')
# df = pd.read_excel('H:/test/新数据集/数据集12.xlsx',sheet_name='数据集12')
data = df.iloc[:, 1:].copy().values
# print(data)


X = data[:,:-1]

feature_cols = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation', 'landUse']

Continuousfeatures = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                      'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation']

X = pd.DataFrame(X, columns=feature_cols)

model_paths = [
    ("LR", "H:/test/LR.pkl"),
    ("SVM", "H:/test/SVM.pkl"),
    ("XGBoost", "H:/test/XGBoost.pkl"),
    ("GBRT", "H:/test/GBRT.pkl"),
    ("RF", "H:/test/RF.pkl")
    ]

for feature in Continuousfeatures:
    X[feature] = (X[feature]-X[feature].mean()) / (X[feature].std())

X_ohe = pd.get_dummies(X, columns=['landUse'])

results = pd.DataFrame()  # 保存所有模型的预测结果的DataFrame

for model_name, model_path in model_paths:
    loaded_model = load(model_path)  # 加载模型

    predictions = loaded_model.predict(X_ohe)  # 使用模型进行预测

    # 将预测结果添加到结果DataFrame的对应列中
    results[model_name] = predictions

# 保存结果到CSV文件
results.to_csv("H:/test/models_result.csv", index=False)