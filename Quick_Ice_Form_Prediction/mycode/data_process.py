"""
函数总结：
readtxt: 读一个样本(字母异常 → 上下行均值）
neighbor_anomaly: 邻近度判据检测断点并修复
data_process: 批量读取，返回 (K, 320, 2)
check_plot: 画轮廓
可直接运行查看结果
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_PATH = Path(__file__).resolve().parent.parent / "train"
SPACE_K = 3.0  # 相邻点间距超过中位数的 3 倍，判为断线


def readtxt(txtname, folder=FILE_PATH):
    """
    读取一个样本,返回X矩阵
    无法转成float的分量记为 NaN,随后用上下行均值补齐
    列数不是 2 的行（空行、.dat 表头）直接跳过
    """
    X = []
    idx = []
    flag = 0
    with open(Path(folder) / txtname, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()  # 空行自然得到空列表
            if len(parts) != 2:   # 空行、表头、异常列数：跳过
                continue
            row = []
            for p in parts:
                try:
                    row.append(float(p))
                except ValueError:
                    flag = 1
                    idx.append(len(X) + 1)
                    row.append(np.nan)
            X.append(row)

    X = np.array(X, dtype=float)
    nr, nc = X.shape
    for j in range(nc):
        col = X[:, j]
        bad = np.where(np.isnan(col))[0]
        col[bad] = (col[bad - 1] + col[bad + 1]) / 2  # 首尾不会有异常

    if flag:
        print(f"将第{sorted(set(idx))}行字母转换为了上下两行均值")
    else:
        print(f"没有异常字母")

    return X


def neighbor_anomaly(X, thresh=None):
    """
    由于x和y分布在翼型附近，故不适合多元高斯分布，这里由于数据量不大，采用临近度的方法
    相邻点间距超过 thresh（默认 3×中位间距）的边视为断边
    两条相邻边都超标的内部点判为异常，只替换跳变的那个分量（所以+1）
    """
    cols = X.shape[1]
    delta = X[1:] - X[:-1]
    dist = np.sqrt(np.sum(delta ** 2, axis=1))
    if thresh is None:
        thresh = SPACE_K * np.median(dist)

    bad = np.where((dist[:-1] > thresh) & (dist[1:] > thresh))[0] + 1  # 巧妙地错开一位，判断bad位置
    if bad.size == 0:
        print(f"没有异常数据")
        return X

    X_rep = X.copy()
    for i in bad:
        print(f"第{i}组数据存在异常，替换为上下文均值")
        for j in range(cols):  # 只补真正跳变的分量
            if abs(delta[i - 1, j]) > thresh and abs(delta[i, j]) > thresh:
                X_rep[i, j] = (X_rep[i - 1, j] + X_rep[i + 1, j]) / 2

    return X_rep


def data_process(name_set, folder=FILE_PATH):
    """批量读取一个目录下的样本，返回 (K, 320, 2) 与温度"""
    T = [int(name.split('_')[-2]) for name in name_set]  # 1*5
    X_full = np.stack([neighbor_anomaly(readtxt(name, folder)) for name in name_set])  # 5*320*2
    return X_full, T


def check_plot(X, save=None):
    """画冰形轮廓；save 给路径则存图，否则弹窗。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(X[:, 0], X[:, 1], marker='.', ms=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_aspect('equal')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    name = "naca_0.5_259_250.txt"
    X = readtxt(name)
    check_plot(X)
    X_rep = neighbor_anomaly(X)
    check_plot(X_rep)
    print(f"修复前后最大差值 = {np.abs(X_rep - X).max():.3g}")
