# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:10:20 2023

@author: pc
"""

from sklearn.ensemble import GradientBoostingClassifier as GBRT
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import optuna
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
import plotly
from optuna.visualization import plot_optimization_history, plot_param_importances

#读取数据
df = pd.read_excel('H:/test/新数据集/数据集11.xlsx',sheet_name='数据集11')
data = df.iloc[:, 1:].copy().values
# print(data)


X = data[:,:-1]
y = data[:,-1]


feature_cols = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation', 'landUse']

Continuousfeatures = ['DEM', 'aspect', 'Avel', 'NDVI', 'road_Dis', 'stream_Dis',
                      'topographic_wetness', 'plan_curv', 'profile_curv', 'slope', 'precipitation']

X = pd.DataFrame(X, columns=feature_cols)



for feature in Continuousfeatures:
    X[feature] = (X[feature]-X[feature].mean()) / (X[feature].std())

X_arr = np.array(X)

X_ohe = pd.get_dummies(X, columns=['landUse'])

feat_labels = X_ohe.columns[:]

#定义搜索空间
def objective(trial, data=X, target=y):
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=42)
    param = {
        
        
        
        #max AUPRC: 0.9432490729934322
        # Best trial: {'n_estimators': 881, 'max_depth': 10, 
        # 'learning_rate': 0.0909856243779888, 'subsample': 0.5807269766341562}
        
        "n_estimators": trial.suggest_int('n_estimators', 1, 1001, 10),
        "max_depth": trial.suggest_int('max_depth', 5, 20),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.1),
        "subsample": trial.suggest_float('subsample', 0.5, 1.0)
        
    }
    model = GBRT(**param)
    y_real = []
    y_proba = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
    for train_index, test_index in cv.split(X_arr, y):
        Xtrain, Xtest = X_arr[train_index], X_arr[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        
        smo = SMOTENC(sampling_strategy=1, categorical_features=[11], random_state=42)
        X_smo, y_smo = smo.fit_resample(Xtrain, ytrain)
        
        Xtest = pd.DataFrame(Xtest, columns=feature_cols)
        
        X_smo = pd.DataFrame(X_smo, columns=feature_cols)
        
        Xtest_ohe = pd.get_dummies(Xtest, columns=['landUse'])
        
        X_smo_ohe = pd.get_dummies(X_smo, columns=['landUse'])
        
        
        model.fit(X_smo_ohe, y_smo)
        
        pred_proba = model.predict_proba(Xtest_ohe)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:, 1])
        y_real.append(ytest)
        y_proba.append(pred_proba[:, 1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba, pos_label=None, sample_weight=None)
    aupr = auc(recall, precision)
    return aupr

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print(f'max AUPRC: {study.best_trial.value}')  # 输出最优结果的AUPR
print('Best trial:', study.best_trial.params)    
optimization_history = plot_optimization_history(study)
param_importances = plot_param_importances(study)
plotly.offline.plot(optimization_history, filename='Optimization_history_plot_GBRT.html')
plotly.offline.plot(param_importances, filename='param_importances_plot_GBRT.html') 