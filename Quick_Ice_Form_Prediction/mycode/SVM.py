import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from data_process import data_process
from POD import POD, inverse_POD

def SVM(T, coef, kernel="rbf", C=10, gamma=0.1, epsilon=1e-3):
    T = np.asarray(T, float).reshape(-1, 1)   # (K, 1)：K 个样本，1 个特征
    sx = StandardScaler().fit(T)
    sy = StandardScaler().fit(coef)
    Xs, Ys = sx.transform(T), sy.transform(coef) # 得到标准化后的数据

    # 对每个模态系数各训练一个SVR
    models = [SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon).fit(Xs,Ys[:,j])
              for j in range(coef.shape[1])]

    return models, sx, sy

def predict(T_new, models, sx, sy):
    Xs = sx.transform(np.array([[T_new]], float))
    a = np.array([m.predict(Xs)[0] for m in models]).reshape(1, -1)

    return sy.inverse_transform(a)[0]

