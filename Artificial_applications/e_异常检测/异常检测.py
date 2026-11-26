import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"
FIG_PATH = Path(__file__).parent / "consequence"


# 估计高斯分布参数
def estimate_gaussian(X):
    # 每个特征单独估计：均值 mu，方差 sigma2（有偏估计，和 PCA.py 一样除以 N）
    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)

    return mu, sigma2


# 计算样本的概率密度
def gaussian_prob(X, mu, sigma2):
    # 各特征相互独立，p(x) 是各维正态密度的连乘
    # 连乘容易下溢，所以取对数再相加
    const = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
    log_p = const - 0.5 * np.sum((X - mu) ** 2 / sigma2, axis=1)

    return np.exp(log_p)


# 计算 F1--调和平均
def compute_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0    # 查准率
    recall = tp / (tp + fn) if tp + fn else 0   # 查全率
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return f1


# 用带标签的验证集选阈值
def select_threshold(yval, pval):
    # 阈值只有落在样本概率之间时判定结果才会变，所以直接拿排好序的概率值当候选
    # 非常聪明的一步！
    eps_list = np.append(np.sort(np.unique(pval)), pval.max() * 1.01)
    pred = [(pval < eps).astype(int) for eps in eps_list]
    f1_list = np.array([compute_f1(yval, y) for y in pred])
    best = int(np.argmax(f1_list))
    return eps_list[best], f1_list[best], eps_list, f1_list


def main(name):
    file = DATA_PATH / name
    mat_dict = loadmat(file)
    print(mat_dict.keys())   # 打印mat里面所有变量名，比如 dict_keys(['X', 'yval'])
    Xval = mat_dict["Xval"]
    yval = mat_dict["yval"].ravel()
    X = mat_dict["X"]

    # 手动修改 X 中某个异常点的坐标，模拟一次明显的发动机异常
    idx = 306
    print("修改前 X[{}] = {}".format(idx, np.round(X[idx], 3)))
    X[idx] = [26.0, 25.0]
    print("修改后 X[{}] = {}".format(idx, np.round(X[idx], 3)))

    # 1. 用不带标签的 X 拟合高斯分布
    mu, sigma2 = estimate_gaussian(X)
    print("mu: ", mu)
    print("sigma2: ", sigma2)

    # 2. 用带标签的 Xval、yval 通过 F1 选阈值
    pval = gaussian_prob(Xval, mu, sigma2)
    eps, f1, eps_list, f1_list = select_threshold(yval, pval)
    print("最佳阈值 epsilon: {:.3e}".format(eps))
    print("对应的 F1: {:.4f}".format(f1))

    # 3. 在 X 上检测异常点
    p = gaussian_prob(X, mu, sigma2)
    outlier = p < eps
    print("X 中判为异常的点数: ", outlier.sum())
    print("异常点索引: ", np.where(outlier)[0])

    plot(X, Xval, yval, mu, sigma2, eps, f1, eps_list, f1_list, outlier, idx)


# 绘图
def plot(X, Xval, yval, mu, sigma2, eps, f1, eps_list, f1_list, outlier, idx):
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
    plt.rcParams['axes.unicode_minus'] = False # 显示负号
    plt.rcParams['axes.spines.top'] = False # 去掉边框
    plt.rcParams['axes.spines.right'] = False
    FIG_PATH.mkdir(exist_ok=True)

    def density(ax, lim):
        # 在给定范围内画 p(x) 的等高线：各维独立，等高线是轴对齐的椭圆
        g1, g2 = np.meshgrid(np.linspace(*lim[0], 200), np.linspace(*lim[1], 200))
        grid = np.c_[g1.ravel(), g2.ravel()]
        Z = gaussian_prob(grid, mu, sigma2).reshape(g1.shape)
        levels = np.exp(np.linspace(np.log(Z.max() * 1e-2), np.log(Z.max()), 8))
        ax.contourf(g1, g2, Z, levels=levels, cmap='Blues', alpha=0.75)
        ax.contour(g1, g2, Z, levels=levels, colors='#2471a3',
                   linewidths=0.6, alpha=0.5)
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])

    def span(P):
        # 画图范围：比数据每边多留 2 个单位
        x_lim = (P[:, 0].min() - 2, P[:, 0].max() + 2)
        y_lim = (P[:, 1].min() - 2, P[:, 1].max() + 2)
        return [x_lim, y_lim]

    def label(ax, title):
        ax.set_xlabel('特征 $x_1$')
        ax.set_ylabel('特征 $x_2$')
        ax.set_title(title)
        ax.legend(loc='upper left', frameon=False)

    # 图1：训练数据 + 拟合的高斯分布
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    density(ax, span(X))
    ax.scatter(X[:, 0], X[:, 1], s=14, color='#34495e', alpha=0.55,
               edgecolors='none', label='样本点')
    ax.scatter(*mu, marker='*', s=220, color='#f7d154', edgecolors='black',
               lw=0.8, zorder=6, label='均值 $\\mu$')
    label(ax, 'X 的散点与拟合的二维高斯密度')
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'anomaly_fit.png', dpi=300)

    # 图2：验证集上的判定结果，命中、漏检、误报分别标出
    flag = gaussian_prob(Xval, mu, sigma2) < eps
    hit = flag & (yval == 1) # 真异常且被检出
    miss = ~flag & (yval == 1) # 真异常但漏检
    false = flag & (yval == 0) # 正常点被误报
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    density(ax, span(Xval))
    ax.scatter(Xval[~flag & ~false, 0], Xval[~flag & ~false, 1], s=14,
               color='#34495e', alpha=0.55, edgecolors='none',
               label='正常点（{}）'.format((~flag & ~false).sum()))
    ax.scatter(Xval[hit, 0], Xval[hit, 1], s=110, facecolors='none',
               edgecolors='#c0392b', lw=1.8, label='已检出（{}）'.format(hit.sum()))
    ax.scatter(Xval[miss, 0], Xval[miss, 1], s=70, marker='D',
               facecolors='none', edgecolors='#e67e22', lw=1.6,
               label='漏检（{}）'.format(miss.sum()))
    ax.scatter(Xval[false, 0], Xval[false, 1], s=90, marker='^',
               facecolors='none', edgecolors='#8e44ad', lw=1.6,
               label='误报（{}）'.format(false.sum()))
    label(ax, '验证集判定结果（$\\epsilon$ = {:.2e}，F1 = {:.3f}）'.format(eps, f1))
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'anomaly_val.png', dpi=300)

    # 图3：F1 随阈值的变化，阈值跨了几十个数量级，所以横轴取 log
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.log10(eps_list), f1_list, color='#5b7db1', lw=2)
    ax.scatter(np.log10(eps), f1, s=80, color='#c0392b', zorder=5, label='最佳阈值')
    ax.annotate('$\\epsilon$ = {:.2e}\nF1 = {:.3f}'.format(eps, f1),
                xy=(np.log10(eps), f1), xytext=(40, -40),
                textcoords='offset points', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#c0392b'))
    ax.set_xlabel('阈值 $\\log_{10}\\epsilon$')
    ax.set_ylabel('F1')
    ax.set_title('验证集上 F1 随阈值的变化')
    ax.grid(alpha=0.25, ls=':')
    ax.legend(loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'anomaly_f1.png', dpi=300)

    # 图4：X 的检测结果，标出异常点
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    density(ax, span(X))
    ax.scatter(X[~outlier, 0], X[~outlier, 1], s=14, color='#34495e',
               alpha=0.55, edgecolors='none', label='正常点')
    ax.scatter(X[outlier, 0], X[outlier, 1], s=130, facecolors='none',
               edgecolors='#c0392b', lw=1.8,
               label='异常点（{}）'.format(outlier.sum()))
    ax.annotate('手动修改的点\n({:.1f}, {:.1f})'.format(*X[idx]),
                xy=X[idx], xytext=(-110, -40), textcoords='offset points',
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    label(ax, 'X 的异常检测结果（$\\epsilon$ = {:.2e}）'.format(eps))
    fig.tight_layout()
    fig.savefig(FIG_PATH / 'anomaly_detection.png', dpi=300)

    plt.show()


if __name__ == "__main__":
    name = "ch8_detection_data1.mat"
    main(name)
