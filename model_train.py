# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:49:56 2023

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:56:49 2022

@author: pc
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix
# import sklearn.tree as st
# import sklearn.ensemble as se
# from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTENC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGB
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from joblib import dump
import os

#读取数据
df = pd.read_excel('H:/test/新数据集/数据集11.xlsx',sheet_name='数据集11')
# df = pd.read_excel('H:/test/新数据集/数据集12.xlsx',sheet_name='数据集12')
data = df.iloc[:, 1:].copy().values
# print(data)


X = data[:,:-1]
y = data[:,-1]


feature_cols = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation', 'landUse']

Continuousfeatures = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                      'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation']

X = pd.DataFrame(X, columns=feature_cols)

# X.drop('Avel', axis=1)


for feature in Continuousfeatures:
    X[feature] = (X[feature]-X[feature].mean()) / (X[feature].std())

# X_ohe.hist(sharex=True)
# plt.show()

X_arr = np.array(X)

X_ohe = pd.get_dummies(X, columns=['landUse'])

feat_labels = X_ohe.columns[:]
# #数据归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# X_minmax = min_max_scaler.fit_transform(X_ohe)
# print(X_train_minmax)

# #划分训练集、测试集
# X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, stratify=y, test_size=0.3, random_state=0)


# #标准化
# st_x = StandardScaler(feature_names_in=['dem', 'asp', 'avel', 'ndvi', 'euc', 'slop'])  
# X_train = st_x.fit_transform(X_train)  
# X_test =  st_x.transform(X_test)


# #SMOTE上采样
# smo = SMOTE(sampling_strategy=1, k_neighbors=10, random_state=42)
# X_smo, y_smo = smo.fit_resample(X_train, y_train)


# #Borderline-SMOTE
# smo = BorderlineSMOTE(sampling_strategy=1, k_neighbors=10, random_state=42)
# X_smo, y_smo = smo.fit_resample(X_train, y_train)

# #随机过采样
# smo = RandomOverSampler(sampling_strategy=1/10, random_state=42)
# X_smo, y_smo = smo.fit_resample(X_train, y_train)

# #SMOTETomek 
# smo = SMOTETomek(sampling_strategy=1, random_state=42)
# X_smo, y_smo = smo.fit_resample(X_train, y_train)



models = [('LR', LogisticRegression(max_iter=1000, C=100, solver= 'liblinear', tol=1e-05, class_weight='balanced')),
          ('SVM', svm.SVC( C=10, gamma=1, probability=True)),
          ('XGBoost', XGB(n_estimators=861, max_depth=20, gamma=0.5, min_child_weight=1, 
                          reg_lambda=0, reg_alpha=0.01,subsample=0.8, 
                          colsample_bytree=0.53, use_label_encoder=False)),
          ('GBRT', GradientBoostingClassifier(n_estimators=881, learning_rate=0.09,
                          max_depth=10, subsample=0.58, random_state=0)),
          ('MLP', MLPClassifier(hidden_layer_sizes=(128,128), activation = 'tanh',
                          learning_rate_init=0.001, max_iter=1000, random_state=1)),
          ('RF', RandomForestClassifier(n_estimators=351, max_depth=20, 
                          min_samples_split=2, min_samples_leaf=1, max_features='log2', 
                          random_state=2))
          ]
save_path = "H:/test/"  
# #二元逻辑回归
# model1 = LogisticRegression()
# model1.fit(X_smo, y_smo)

# y_pred = model1.predict(X_train)
# y_pred_proba = model1.predict_proba(X_train)

# # #SVM参数优化（网格搜索）
# # def svm_cross_validation(train_x, train_y):    
# #     from sklearn.model_selection import GridSearchCV    
# #     from sklearn.svm import SVC    
# #     model = SVC(kernel='rbf', probability=True)    
# #     param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
# #     grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
# #     grid_search.fit(train_x, train_y)    
# #     best_parameters = grid_search.best_estimator_.get_params()    
# #     for para, val in list(best_parameters.items()):    
# #         print(para, val)    
# #     model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
# #     model.fit(train_x, train_y)    
# #     return model
# #SVM
# clf = svm.SVC(probability=True)
# clf.fit(X_smo, y_smo)
# y_pred2 = clf.predict(X_train)
# y_pred_proba2 = clf.predict_proba(X_train)

# # clf = svm_cross_validation(X_smo, y_smo)
# # y_pred2 = clf.predict(X_test)
# # y_pred_proba2 = clf.predict_proba(X_test)

# #随机森林
# forest = RandomForestClassifier()
# forest.fit(X_smo, y_smo)
# y_pred3 = forest.predict(X_train)
# y_pred_proba3 = forest.predict_proba(X_train)

# #XGBoost
# reg = XGBR(use_label_encoder=False).fit(X_smo, y_smo)
# y_pred4 = reg.predict(X_train)
# y_pred_proba4 = reg.predict_proba(X_train)


for name, model in models:
    
    y_real = []
    y_proba = []
    params = []
    # y_proba_all = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    # print(type(cv.split(X_arr, y)))
    # k_fold = KFold(n_splits=5, shuffle=True, random_state=12345)
    for i, (train_index, test_index) in enumerate(cv.split(X_arr, y)):
        Xtrain, Xtest = X_arr[train_index], X_arr[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        
                
        smo = SMOTENC(sampling_strategy=1, categorical_features=[11], random_state=42)
        X_smo, y_smo = smo.fit_resample(Xtrain, ytrain)
        
        # if i==3:
            
        #     df_x = pd.DataFrame(X_smo)
        #     df_y = pd.DataFrame(y_smo)
        #     df_x.to_csv('H:/test/GBRT五折交叉实验的特征相关性排序/X_smo.csv',index= False, header= False)
        #     df_y.to_csv('H:/test/GBRT五折交叉实验的特征相关性排序/y_smo.csv',index= False, header= False)
        
        Xtest = pd.DataFrame(Xtest, columns=feature_cols)
        
        X_smo = pd.DataFrame(X_smo, columns=feature_cols)
        
        Xtest_ohe = pd.get_dummies(Xtest, columns=['landUse'])
        
        X_smo_ohe = pd.get_dummies(X_smo, columns=['landUse'])
        
        
        model.fit(X_smo_ohe, y_smo)
        
       
        filename = f"{name}.pkl"  # 根据模型名称和时间戳构建文件名
        file_path = os.path.join(save_path, filename)  # 构建完整的文件路径

        dump(model, file_path)  # 保存模型到文件
        
        # #保存模型参数
        # if name == 'LR':
        #     LR_params = model.coef_
        #     # params.append(LR_params)
        #     print()
        # elif name == 'SVM':
        #     SVM_params = model.support_vectors_
        #     params.append(SVM_params)
        # elif name == 'MLP':
        #     MLP_params = model.coefs_
        #     params.append(MLP_params)
        # elif name == 'GBRT':
        #     GBRT_params = model.estimators_
        #     params.append(GBRT_params)
        # elif name == 'RF':
        #     RF_params = model.estimators_
        #     params.append(RF_params)
            
        
        #测试集
        y_pred = model.predict(Xtest_ohe)
        pred_proba = model.predict_proba(Xtest_ohe)
        # precision, recall, _ = precision_recall_curve(ytest, pred_proba[:, 1])
        y_real.append(ytest)
        y_proba.append(pred_proba[:, 1])
        
        # #训练集
        # pred_proba = model.predict_proba(X_smo_ohe)
        # # precision, recall, _ = precision_recall_curve(y_smo, pred_proba[:, 1])
        # y_real.append(y_smo)
        # y_proba.append(pred_proba[:, 1])
        
        # 输出特征重要性（RF）
        # if name == 'RF':
        #     pred_proba_all = model.predict_proba(X_ohe)
        #     y_proba_all.append(pred_proba_all[:, 1])
        #     importances = model.feature_importances_
        #     indices = np.argsort(importances)[::-1]
        #     for f in range(X_smo_ohe.shape[1]):
        #         print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
            
        #每一折交叉实验中模型的precision，recall， f1-score
        precision = precision_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)
        print(name, " Precision:", precision, " Recall:" , recall, "F1_score", f1)
        
        if name == 'GBRT':
            print("Model:", name)
            print(model.feature_importances_)
            
        # if name == 'XGBoost':
        #     # 获取特征重要性得分
        #     importance = model.get_score(importance_type='weight')

        #     # 可视化特征重要性
        #     model.plot_importance(importance)                
    # params = np.concatenate(params)
    # print(params)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba, pos_label=None, sample_weight=None)
    aupr = round(auc(recall, precision), 2)#计算面积的
    plt.plot(recall, precision, label=name+'(AUPRC = %0.2f)' % aupr)
    

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.plot([0, 1], [1, 0], color='navy', label='Reference line', linestyle='--')
plt.xlabel('Recall', fontproperties='Times New Roman')
plt.ylabel('Precision', fontproperties='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'weight': 'normal'}, loc='lower left', fontsize='small')
plt.show()



# m = confusion_matrix(y_test, y_pred)
# print(m)

# auc = roc_auc_score(y_test, y_pred_proba[:,1])
# print(auc)

# f1 = classification_report(y_test, y_pred)
# print(f1)

# #precision
# print("prec_LR:", precision_score(y_test, y_pred))
# print("prec_SVM:", precision_score(y_test, y_pred2))
# print("prec_RF:", precision_score(y_test, y_pred3))
# print("prec_XGBoost:", precision_score(y_test, y_pred4))

# #混淆矩阵
# print("confusion_matrix_LR:", confusion_matrix(y_test, y_pred))
# print("confusion_matrix_SVM:", confusion_matrix(y_test, y_pred2))
# print("confusion_matrix_RF:", confusion_matrix(y_test, y_pred3))
# print("confusion_matrix_XGBoost:", confusion_matrix(y_test, y_pred4))

# #recall
# print("recall_LR:", recall_score(y_test, y_pred))
# print("recall_SVM:", recall_score(y_test, y_pred2))
# print("recall_RF:", recall_score(y_test, y_pred3))
# print("recall_XGBoost:", recall_score(y_test, y_pred4))

# #AUC
# print("auc_LR:", roc_auc_score(y_test, y_pred_proba[:,1]))
# print("auc_SVM:", roc_auc_score(y_test, y_pred_proba2[:,1]))
# print("auc_RF:", roc_auc_score(y_test, y_pred_proba3[:,1]))
# print("auc_XGBoost:", roc_auc_score(y_test, y_pred_proba4[:,1]))

# #F1-score
# print("f1score_LR:", f1_score(y_test, y_pred))
# print("f1score_SVM:", f1_score(y_test, y_pred2))
# print("f1score_RF:", f1_score(y_test, y_pred3))
# print("f1score_XGBoost:", f1_score(y_test, y_pred4))

# #ROC曲线
# fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1])
# fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred_proba2[:, 1])
# fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred_proba3[:, 1])
# fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred_proba4[:, 1])
# plt.plot(fpr, tpr, label='LR')
# plt.plot(fpr2, tpr2, color='darkorange', label='SVM')
# plt.plot(fpr3, tpr3, color='red', label='RF')
# plt.plot(fpr4, tpr4, color='black', label='XGBoost')
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.show()


