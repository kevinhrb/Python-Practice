import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"
FIG_PATH = Path(__file__).parent / "consequence"

def main(name):
    path = DATA_PATH / name
    # 这里数据很干净，直接用np.loadtxt
    X = np.loadtxt(path)
    # 标准化
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_std = (X - mu) / sigma

    # SVD
    U, S, Vt = np.linalg.svd(X_std)
    N = X_std.shape[0]
    eigval = S ** 2 / N # 这里是一个有偏估计，与上文保持一致，计算机领域默认除以N？
    eigvec = Vt
    
    # 因为只有两列，所以K取1（其余特征值必为0）
    k = 1
    X_pca = X_std @ Vt.T[:, :k]

    print("eigen value: ", eigval)
    print("eigen vector: ", eigvec)
    print("降维后 X_pca shape: ", X_pca.shape)
    print("第一主成分方差贡献率: {:.2%}".format(eigval[0] / eigval.sum()))

    plot(X, X_pca, mu, sigma, eigval, Vt)


# 绘图
def plot(X, X_pca, mu, sigma, eigval, Vt):
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
    plt.rcParams['axes.unicode_minus'] = False # 显示负号
    plt.rcParams['axes.spines.top'] = False # 去掉上、右边框
    plt.rcParams['axes.spines.right'] = False
    FIG_PATH.mkdir(exist_ok=True)

    # Vt 的行是标准化空间里的单位方向向量，逐维乘 sigma 就换回原始坐标下的方向
    # 投影坐标的方差是特征值，所以箭头长度取 2 倍标准差：2*sqrt(特征值)*方向
    direction = Vt * sigma
    vec = 2 * np.sqrt(eigval)[:, None] * direction
    color = ['#c0392b', '#2471a3']
    label = ['第一主成分 $v_1$', '第二主成分 $v_2$']

    # 图1：原始数据 + 主/次向量方向
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(X[:, 0], X[:, 1], s=16, color='#5b7db1', alpha=0.45,
               edgecolors='none', label='样本点')
    ax.scatter(*mu, s=110, color='#f7d154', edgecolors='black', linewidths=1.2,
               zorder=5, label='样本均值')
    for i in range(2):
        for sign, ls in [(1, '-'), (-1, '--')]:
            ax.annotate('', xy=mu + sign * vec[i], xytext=mu,
                        arrowprops=dict(arrowstyle='-|>', color=color[i], lw=2, ls=ls,
                                        shrinkA=0, shrinkB=0, mutation_scale=16))
        ax.text(*(mu + 1.35 * vec[i]), label[i], color=color[i], fontsize=11,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color[i], lw=0.6, alpha=0.9))
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('原始数据与主成分方向（箭头为 2 倍标准差，虚线为反方向）')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'pca_original_axes.png', dpi=300)

    # 图2：降维后的一维投影分布
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(X_pca[:, 0], bins=24, color='#5b7db1', edgecolor='white', alpha=0.9)
    ax.axvline(0, color='#c0392b', lw=1.2, ls='--', label='投影原点（样本均值）')
    # 投影坐标的均值是0、方差是特征值，用 N(0, λ1) 的密度乘样本数和组距做对照
    z = np.linspace(X_pca.min(), X_pca.max(), 200)
    bin_w = np.ptp(X_pca) / 24
    ax.plot(z, len(X_pca) * bin_w * np.exp(-z ** 2 / (2 * eigval[0])) / np.sqrt(2 * np.pi * eigval[0]),
            color='#c0392b', lw=1.4, label=r'$N(0,\ \lambda_1)$ 拟合密度')
    ax.set_xlabel('第一主成分得分 $t_1$')
    ax.set_ylabel('样本数量')
    ax.set_title('降维后的一维投影分布（方差贡献率 {:.2%}）'.format(eigval[0] / eigval.sum()))
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.25, ls=':')
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'pca_projection_hist.png', dpi=300)

    # 图3：降维的几何意义——样本沿 v1 投影，残差被丢弃
    t = X_pca[:, 0]
    proj = mu + t[:, None] * direction[0] # 投影点在原始坐标下的位置
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(X[:, 0], X[:, 1], s=16, color='#95a5a6', alpha=0.5,
               edgecolors='none', label='样本点')
    ax.plot(proj[:, 0], proj[:, 1], 'o', ms=2.5, color='#c0392b',
            alpha=0.8, label='一维投影点')
    axis = mu + np.array([t.min(), t.max()])[:, None] * direction[0]
    ax.plot(axis[:, 0], axis[:, 1], color='#c0392b', lw=1.6, label='第一主成分轴')
    for i in range(0, len(X), 5): # 每 5 个点画一条残差线，避免糊成一片
        ax.plot([X[i, 0], proj[i, 0]], [X[i, 1], proj[i, 1]],
                color='#7f8c8d', lw=0.6, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('样本沿第一主成分投影（灰线为被丢弃的残差）')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'pca_projection_geometry.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    name = "ch7_pca_data1.txt"
    main(name)
