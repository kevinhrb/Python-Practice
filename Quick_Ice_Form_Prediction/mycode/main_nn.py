"""主程序（神经网络）：训练 → 预测测试集 → 误差统计，并与 SVR 对比。"""
import numpy as np
from pathlib import Path

from data_process import data_process, readtxt
from POD import POD, inverse_POD
from SVM import SVM, predict as svm_predict
from NN import NN, predict as nn_predict
from post_process import errors, plot_compare, write_dat

TEST_DIR = Path(__file__).resolve().parent.parent / "test"
OUT_DIR = Path(__file__).resolve().parent / "pred_nn"
FIG_DIR = Path(__file__).resolve().parent / "figs_nn"
DAT_DIR = Path(__file__).resolve().parent / "dat_nn"
K = 3


def main():
    name_set_train = [f"naca_0.5_{t}_250.txt" for t in (257, 259, 261, 263, 265)]
    name_set_test = [f"naca_0.5_{t}_250.dat" for t in (258, 260, 264)]

    X_train, T_train = data_process(name_set_train)
    coef, mean, modes = POD(X_train, K)

    nn_model, sx_nn, sy_nn = NN(T_train, coef, seed=42)
    svm_models, sx_svm, sy_svm = SVM(T_train, coef)

    X_test, T_test = data_process(name_set_test, TEST_DIR)
    mean_shape = mean.reshape(320, 2)
    X_wing = readtxt("wing.dat")

    for folder in (OUT_DIR, FIG_DIR, DAT_DIR):
        folder.mkdir(exist_ok=True)

    print(f"\n{'T':>5} | {'NN MAE':>9} {'NN RMSE':>9} {'NN Max':>9} {'NN RelL2':>8}"
          f" | {'SVM MAE':>9} {'SVM RMSE':>9} {'SVM Max':>9} {'SVM RelL2':>9} {'基线Max':>9}")
    print("-" * 110)

    rows_nn, rows_svm = [], []
    for i, t in enumerate(T_test):
        a_nn = nn_predict(t, nn_model, sx_nn, sy_nn)
        X_nn = inverse_POD(a_nn[None], mean, modes)[0]

        a_svm = svm_predict(t, svm_models, sx_svm, sy_svm)
        X_svm = inverse_POD(a_svm[None], mean, modes)[0]

        e_nn = errors(X_test[i], X_nn)
        e_svm = errors(X_test[i], X_svm)
        base = np.abs(X_test[i] - mean_shape).max()
        rows_nn.append(e_nn)
        rows_svm.append(e_svm)
        print(f"{t:>5} | {e_nn[0]:>9.2e} {e_nn[1]:>9.2e} {e_nn[2]:>9.2e} {e_nn[3]:>8.2%}"
              f" | {e_svm[0]:>9.2e} {e_svm[1]:>9.2e} {e_svm[2]:>9.2e} {e_svm[3]:>9.2%} {base:>9.2e}")

        np.savetxt(OUT_DIR / f"pred_nn_{t}.txt", X_nn, fmt="%.6e")
        plot_compare(X_test[i], X_nn, X_wing, t, FIG_DIR / f"compare_nn_{t}.png")
        write_dat(DAT_DIR / f"ice_nn_{t}.dat", X_test[i], X_nn, t)

    m_nn = np.mean(np.array(rows_nn), axis=0)
    m_svm = np.mean(np.array(rows_svm), axis=0)
    print("-" * 110)
    print(f"{'平均':>5} | {m_nn[0]:>9.2e} {m_nn[1]:>9.2e} {m_nn[2]:>9.2e} {m_nn[3]:>8.2%}"
          f" | {m_svm[0]:>9.2e} {m_svm[1]:>9.2e} {m_svm[2]:>9.2e} {m_svm[3]:>9.2%}")
    print(f"\n图  -> {FIG_DIR}")
    print(f"dat -> {DAT_DIR}")


if __name__ == '__main__':
    main()
