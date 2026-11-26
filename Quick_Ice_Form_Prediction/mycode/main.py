import numpy as np
from pathlib import Path

from data_process import data_process, readtxt
from POD import POD, inverse_POD
from SVM import SVM, predict
from post_process import errors, plot_compare, write_dat

TEST_DIR = Path(__file__).resolve().parent.parent / "test"
OUT_DIR = Path(__file__).resolve().parent / "pred"
FIG_DIR = Path(__file__).resolve().parent/ "figs_SVM"
DAT_DIR = Path(__file__).resolve().parent / "dat"
K = 3

def main():
    name_set_train = [f"naca_0.5_{t}_250.txt" for t in (257, 259, 261, 263, 265)]
    name_set_test = [f"naca_0.5_{t}_250.dat" for t in (258, 260, 264)]

    X_train, T_train = data_process(name_set_train)
    coef, mean, modes = POD(X_train, K)
    models, sx, sy = SVM(T_train, coef)

    X_test, T_test = data_process(name_set_test, TEST_DIR)
    mean_shape = mean.reshape(320, 2)
    X_wing = readtxt("wing.dat")          # 干净翼型，用作对比基准

    for folder in (OUT_DIR, FIG_DIR, DAT_DIR):
        folder.mkdir(exist_ok=True)

    print(f"\n{'T':>5} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10} {'RelL2':>8} {'基线Max':>10}")

    rows = []
    for i, t in enumerate(T_test):
        a = predict(t, models, sx, sy)
        X_pred = inverse_POD(a[None], mean, modes)[0]
        mae, rmse, mx, rel = errors(X_test[i], X_pred)
        base = np.abs(X_test[i] - mean_shape).max()
        rows.append((mae, rmse, mx, rel))
        print(f"{t:>5} {mae:>10.3e} {rmse:>10.3e} {mx:>10.3e} {rel:>8.2%} {base:>10.3e}")

        np.savetxt(OUT_DIR / f"pred_{t}.txt", X_pred, fmt="%.6e")
        plot_compare(X_test[i], X_pred, X_wing, t, FIG_DIR / f"compare_{t}.png")
        write_dat(DAT_DIR / f"ice_{t}.dat", X_test[i], X_pred, t)

    mae, rmse, mx, rel = np.mean(np.array(rows), axis=0)
    print(f"{'平均':>5} {mae:>10.3e} {rmse:>10.3e} {mx:>10.3e} {rel:>8.2%}")
    print(f"\n图  -> {FIG_DIR}")
    print(f"dat -> {DAT_DIR}")


if __name__ == '__main__':
    main()
