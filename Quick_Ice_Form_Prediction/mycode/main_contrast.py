"""SVM 自对比：核函数固定 rbf，扫描 C 与 epsilon 的 6 种组合，出表 + 出图。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data_process import data_process, readtxt
from POD import POD, inverse_POD
from SVM import SVM, predict
from NN import NN, predict as nn_predict
from post_process import errors

TEST_DIR = Path(__file__).resolve().parent.parent / "test"
FIG_DIR = Path(__file__).resolve().parent / "figs_contrast"
K = 3

C_LIST = [0.1, 1, 100]
EPS_LIST = [1e-4, 1e-2]        # 即 10^-4、10^-2；若要用字面值 10e-4=1e-3，改这里
GAMMA = 0.1                    # 核函数固定，gamma 固定


def main():
    name_set_train = [f"naca_0.5_{t}_250.txt" for t in (257, 259, 261, 263, 265)]
    name_set_test = [f"naca_0.5_{t}_250.dat" for t in (258, 260, 264)]

    X_train, T_train = data_process(name_set_train)
    coef, mean, modes = POD(X_train, K)

    X_test, T_test = data_process(name_set_test, TEST_DIR)
    FIG_DIR.mkdir(exist_ok=True)

    results = []                                   # (C, eps, MAE, RMSE, Max)
    print(f"\n{'C':>6} {'epsilon':>9} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10}")
    print("-" * 50)

    for C in C_LIST:
        for eps in EPS_LIST:
            models, sx, sy = SVM(T_train, coef, C=C, gamma=GAMMA, epsilon=eps)
            rows = []
            for i, t in enumerate(T_test):
                a = predict(t, models, sx, sy)
                X_pred = inverse_POD(a[None], mean, modes)[0]
                rows.append(errors(X_test[i], X_pred))
            mae, rmse, mx = np.mean(np.array(rows)[:, :3], axis=0)
            results.append((C, eps, mae, rmse, mx))
            print(f"{C:>6g} {eps:>9g} {mae:>10.3e} {rmse:>10.3e} {mx:>10.3e}")

    best = min(results, key=lambda r: r[3])
    print("-" * 50)
    print(f"最小 RMSE：C={best[0]:g}, epsilon={best[1]:g}  (RMSE={best[3]:.3e})")

    # ---- 绘图：两组柱状图，分别看 RMSE 和最大误差 ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.35
    x = np.arange(len(C_LIST))

    for k, eps in enumerate(EPS_LIST):
        rmse = [r[3] for r in results if r[1] == eps]
        mx = [r[4] for r in results if r[1] == eps]
        off = (k - 0.5) * width
        b1 = ax[0].bar(x + off, rmse, width, label=f'epsilon={eps:g}')
        b2 = ax[1].bar(x + off, mx, width, label=f'epsilon={eps:g}')
        ax[0].bar_label(b1, fmt='%.1e', fontsize=7)
        ax[1].bar_label(b2, fmt='%.1e', fontsize=7)

    for a, title, ylabel in ((ax[0], 'RMSE (mean over 3 test cases)', 'RMSE'),
                             (ax[1], 'Max error (mean over 3 test cases)', 'max |pred - true|')):
        a.set_xticks(x)
        a.set_xticklabels([f'C={c:g}' for c in C_LIST])
        a.set_title(title)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.3, axis='y')
        a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "svm_contrast.png", dpi=200)
    plt.close(fig)
    print(f"\n图 -> {FIG_DIR / 'svm_contrast.png'}")

    # ---- 三个测试温度：真实冰形 vs SVM vs MLP ----
    svm_models, sx_svm, sy_svm = SVM(T_train, coef, C=10, gamma=GAMMA, epsilon=1e-3)
    nn_model, sx_nn, sy_nn = NN(T_train, coef, seed=42)
    X_wing = readtxt("wing.dat")

    print()
    for i, t in enumerate(T_test):
        X_svm = inverse_POD(predict(t, svm_models, sx_svm, sy_svm)[None], mean, modes)[0]
        X_nn = inverse_POD(nn_predict(t, nn_model, sx_nn, sy_nn)[None], mean, modes)[0]

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.plot(X_wing[:, 0], X_wing[:, 1], '-', color='gray', lw=1.0, label='wing')
        ax.plot(X_test[i][:, 0], X_test[i][:, 1], '-', color='k', lw=1.3, label='true')
        ax.plot(X_svm[:, 0], X_svm[:, 1], '-', color='#2E6DB4', lw=1.8,
                alpha=0.85, label='SVM')
        ax.plot(X_nn[:, 0], X_nn[:, 1], '--', color='#D9731A', lw=1.6, label='MLP')
        ax.set_xlim(-0.02, 0.06)
        ax.set_ylim(-0.04, 0.04)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'T = {t} K')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        ax.text(0.02, 0.96,
                'max err  SVM %.2e / MLP %.2e'
                % (np.abs(X_svm - X_test[i]).max(), np.abs(X_nn - X_test[i]).max()),
                transform=ax.transAxes, fontsize=8, va='top')

        ins = ax.inset_axes([0.58, 0.50, 0.40, 0.44])
        ins.plot(X_wing[:, 0], X_wing[:, 1], '-', color='gray', lw=0.9)
        ins.plot(X_test[i][:, 0], X_test[i][:, 1], '-', color='k', lw=1.2)
        ins.plot(X_svm[:, 0], X_svm[:, 1], '-', color='#2E6DB4', lw=1.6)
        ins.plot(X_nn[:, 0], X_nn[:, 1], '--', color='#D9731A', lw=1.4)
        ins.set_xlim(-0.012, 0.004)
        ins.set_ylim(-0.004, 0.012)
        ins.set_xticks([-0.010, -0.005, 0.000])
        ins.set_yticks([-0.004, 0.000, 0.004, 0.008, 0.012])
        ins.tick_params(labelsize=7)
        ins.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"three_models_{t}.png", dpi=200)
        plt.close(fig)
        print(f"图 -> {FIG_DIR / ('three_models_%d.png' % t)}")


if __name__ == '__main__':
    main()
