"""
POD / PCA 降维：把 (K, 320, 2) 的冰形压成 k 个模态系数。
"""
import numpy as np

from data_process import data_process

def select_a_k(max_k):
    while True:
        s = input(f"输入要保留的模态阶数(k:1~{max_k})：").strip()
        if s.isdigit() and 1 <= int(s) <= max_k:
            return int(s)
        print(f"输入不合法，请输入1~{max_k}范围的整数")


def POD(X, k=None):
    X = X.reshape(len(X), -1)   # 自动对齐为（K，640）
    max_K = len(X) - 1  # 中心化之后，K个样本加起来为0，所以天然线性相关，最多只能张成K-1维

    mean = X.mean(axis=0)   # 中心化，突出冰形特性
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    lam = S ** 2 / max(len(X)-1,1)
    ratio = S ** 2/ np.sum(S ** 2)
    cum = np.cumsum(ratio)

    print("模态 特征值      能量占比    累计占比")
    for i in range(len(S)):
        print("%3d %.4e %8.2f%% %9.2f%%" 
              % (i+1, lam[i], ratio[i]*100, cum[i]*100))
    
    if k is None:
        k = select_a_k(max_K)

    modes = Vt[:k] # 特征向量(3,640)
    coef = U[:,:k] * S[:k] # 降维后数据

    print(f"保留{k}个模态，累计能量{cum[k-1] * 100:2.2f}%")
    return coef, mean, modes


def inverse_POD(coef, mean, modes, n_points=320):
    """还原：coef @ modes + mean，返回 (K, n_points, 2)。"""
    X = coef @ modes + mean
    return X.reshape(len(coef), n_points, 2)


if __name__ == '__main__':
    name_set = [f"naca_0.5_{t}_250.txt" for t in (257, 259, 261, 263, 265)]
    X, T = data_process(name_set)

    coef, mean, modes = POD(X)
    X_hat = inverse_POD(coef, mean, modes)

    print(f"原始 {X.shape} -> 系数 {coef.shape}")
    print(f"重构最大误差: {np.abs(X - X_hat).max():.3g}")
