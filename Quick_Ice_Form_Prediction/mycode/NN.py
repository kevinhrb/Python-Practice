import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def NN(T, coef, hidden=(16, 16), alpha=1e-3, max_iter=5000, seed=42):
    """
    训练一个多输出 MLP：温度 → 全部模态系数（一次输出 k 个）。

    T     : (K,)   温度
    coef  : (K, k) 模态系数
    返回   model, sx, sy
    """
    T = np.asarray(T, float).reshape(-1, 1)
    sx = StandardScaler().fit(T)          # 温度标准化：MLP 对输入尺度敏感
    sy = StandardScaler().fit(coef)       # 系数标准化：输出量级只有 1e-3
    Xs, Ys = sx.transform(T), sy.transform(coef)

    model = MLPRegressor(hidden_layer_sizes=hidden,
                         activation='tanh',
                         solver='lbfgs',
                         alpha=alpha,         # L2 正则
                         max_iter=max_iter,
                         random_state=seed)
    model.fit(Xs, Ys)
    return model, sx, sy


def predict(T_new, model, sx, sy):
    """单个温度 → 长度为 k 的系数向量。"""
    Xs = sx.transform(np.array([[T_new]], float))
    a = model.predict(Xs).reshape(1, -1)
    return sy.inverse_transform(a)[0]
