"""
后处理：
errors： 误差指标（MAE / RMSE / 最大误差 / 相对 L2）
plot_compare： 真值 vs 预测 对比图
write_dat： 写 Tecplot .dat（真实冰形 + 预测冰形，各一个 ZONE）
"""
import numpy as np
import matplotlib.pyplot as plt


def errors(y_true, y_pred):
    e = y_pred - y_true
    return (np.abs(e).mean(),                            # MAE
            np.sqrt((e ** 2).mean()),                    # RMSE
            np.abs(e).max(),                             # 最大误差
            np.linalg.norm(e) / np.linalg.norm(y_true))  # 相对 L2


def plot_compare(X_true, X_pred, X_wing, t, path):
    """前缘放大（x 轴拉伸）+ 逐点误差。叠翼型线，不画平均冰形。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5),
                           gridspec_kw={'width_ratios': [1.6, 1]})

    ax[0].plot(X_wing[:, 0], X_wing[:, 1], '-', color='gray', lw=1.2, label='wing')
    ax[0].plot(X_true[:, 0], X_true[:, 1], 'k-', lw=1.8, label='true')
    ax[0].plot(X_pred[:, 0], X_pred[:, 1], 'r--', lw=1.8, label='pred')
    ax[0].set_xlim(-0.02, 0.06)
    ax[0].set_ylim(-0.04, 0.04)
    ax[0].set_title(f'leading edge   T = {t} K')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_aspect('auto')      # 不锁等比例 → x 轴被拉长，细节更清楚

    err = np.linalg.norm(X_pred - X_true, axis=1)
    ax[1].plot(err, 'b-', lw=1.2)
    ax[1].set_title('pointwise error')
    ax[1].set_xlabel('point index')
    ax[1].set_ylabel('|pred - true|')
    ax[1].set_ylim(bottom=0)

    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_dat(path, X_true, X_pred, T):
    """写 Tecplot 点格式 .dat：两个 ZONE（true / pred），可直接叠图对比。"""
    with open(path, "w", encoding="ascii") as f:
        f.write('VARIABLES = "X", "Y"\n')
        for tag, X in (("true", X_true), ("pred", X_pred)):
            f.write('ZONE T="%s %dk", I=%d , ZONETYPE=Ordered, DATAPACKING=POINT\n'
                    % (tag, T, len(X)))
            for x, y in X:
                f.write("%.6e %.6e\n" % (x, y))
