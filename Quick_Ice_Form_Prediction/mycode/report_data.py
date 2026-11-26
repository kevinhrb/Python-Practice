"""报告数据与图表：计算全部指标，生成 figs_report/，写出 report_numbers.json。"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

from data_process import data_process, readtxt
from POD import POD, inverse_POD
from SVM import SVM, predict as svm_predict
from NN import NN, predict as nn_predict
from post_process import errors

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT.parent / "test"
FIG = ROOT / "figs_report"
FIG.mkdir(exist_ok=True)

K = 3
C_SVM, GAMMA, EPS = 10, 0.1, 1e-3
C_LIST, EPS_LIST = [0.1, 1, 100], [1e-4, 1e-2]

names_train = [f"naca_0.5_{t}_250.txt" for t in (257, 259, 261, 263, 265)]
names_test = [f"naca_0.5_{t}_250.dat" for t in (258, 260, 264)]

X_train, T_train = data_process(names_train)
coef, mean, modes = POD(X_train, K)
X_test, T_test = data_process(names_test, TEST_DIR)
mean_shape = mean.reshape(320, 2)
X_wing = readtxt("wing.dat")

# ---------------- POD 指标 ----------------
A = X_train.reshape(len(X_train), -1)
U, S, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
lam = S ** 2 / (len(A) - 1)
ratio = lam / lam.sum()
cum = np.cumsum(ratio)
X_hat = inverse_POD(coef, mean, modes)
recon_max = float(np.abs(X_train - X_hat).max())

# ---------------- 默认 SVM / MLP ----------------
svm_models, sx_s, sy_s = SVM(T_train, coef, C=C_SVM, gamma=GAMMA, epsilon=EPS)
nn_model, sx_n, sy_n = NN(T_train, coef, seed=42)

tab = {"test_T": [], "baseline_max": [], "baseline_rmse": [],
       "svm": [], "nn": []}
for i, t in enumerate(T_test):
    Xs = inverse_POD(svm_predict(t, svm_models, sx_s, sy_s)[None], mean, modes)[0]
    Xn = inverse_POD(nn_predict(t, nn_model, sx_n, sy_n)[None], mean, modes)[0]
    tab["test_T"].append(int(t))
    tab["baseline_max"].append(float(np.abs(X_test[i] - mean_shape).max()))
    tab["baseline_rmse"].append(float(np.sqrt(((X_test[i] - mean_shape) ** 2).mean())))
    tab["svm"].append([float(v) for v in errors(X_test[i], Xs)])
    tab["nn"].append([float(v) for v in errors(X_test[i], Xn)])

svm_avg = np.mean(tab["svm"], axis=0)
nn_avg = np.mean(tab["nn"], axis=0)
base_max = float(np.mean(tab["baseline_max"]))
base_rmse = float(np.mean(tab["baseline_rmse"]))

# ---------------- 留一验证 ----------------
def loo(make, call):
    out = []
    for i in range(len(T_train)):
        m = np.ones(len(T_train), bool)
        m[i] = False
        obj = make(np.array(T_train)[m], coef[m])
        a = call(T_train[i], *obj)
        Xp = inverse_POD(a[None], mean, modes)[0]
        out.append(errors(X_train[i], Xp))
    return np.mean(np.array(out), axis=0)

svm_loo = loo(lambda T, C: SVM(T, C, C=C_SVM, gamma=GAMMA, epsilon=EPS),
              lambda t, *o: svm_predict(t, *o))
nn_loo = loo(lambda T, C: NN(T, C, seed=42),
             lambda t, *o: nn_predict(t, *o))

# ---------------- 参数自对比 ----------------
contrast = []
for C in C_LIST:
    for eps in EPS_LIST:
        mo, sx_, sy_ = SVM(T_train, coef, C=C, gamma=GAMMA, epsilon=eps)
        rows = []
        for i, t in enumerate(T_test):
            Xp = inverse_POD(svm_predict(t, mo, sx_, sy_)[None], mean, modes)[0]
            rows.append(errors(X_test[i], Xp))
        m = np.mean(np.array(rows)[:, :3], axis=0)
        contrast.append([C, eps, float(m[0]), float(m[1]), float(m[2])])

# ---------------- 预测耗时 ----------------
t0 = time.perf_counter()
for _ in range(200):
    a = svm_predict(258, svm_models, sx_s, sy_s)
    inverse_POD(a[None], mean, modes)
t_pred = (time.perf_counter() - t0) / 200 * 1e3  # ms

# ---------------- 图 1：模态能量 ----------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
idx = np.arange(1, len(S) + 1)
bar = ax.bar(idx, ratio * 100, color='#4C72B0', width=0.6, label='energy ratio')
ax.plot(idx, cum * 100, 'o-', color='#C44E52', lw=1.5, ms=4, label='cumulative')
for x, y in zip(idx, ratio * 100):
    if y > 0.5:
        ax.text(x, y + 1.5, f'{y:.2f}%', ha='center', fontsize=8)
ax.axhline(95, ls='--', lw=1, color='gray')
ax.set_xticks(idx)
ax.set_xlabel('mode')
ax.set_ylabel('energy ratio / %')
ax.set_ylim(0, 110)
ax.legend(fontsize=8, loc='center right')
ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(FIG / "energy.png", dpi=200)
plt.close(fig)

# ---------------- 图 2：SVM 三个误差 ----------------
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
labels = [str(t) + ' K' for t in tab["test_T"]]
svm_arr = np.array(tab["svm"])
for j, (name, scale) in enumerate([('MAE', 1e5), ('RMSE', 1e5), ('MaxErr', 1e3)]):
    vals = svm_arr[:, j] * scale
    b = ax[j].bar(labels, vals, color='#4C72B0', width=0.55)
    ax[j].bar_label(b, fmt='%.2f', fontsize=7)
    ax[j].set_title(f'{name}  ($\\times10^{{{int(np.log10(scale))}}}$)'
                    if scale in (1e3, 1e5) else name, fontsize=9)
    ax[j].grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(FIG / "svm_err.png", dpi=200)
plt.close(fig)

# ---------------- 图 3：SVM vs MLP ----------------
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
x = np.arange(len(labels) + 1)
w = 0.36
svm_rmse = np.append(svm_arr[:, 1], svm_avg[1]) * 1e5
nn_rmse = np.append(np.array(tab["nn"])[:, 1], nn_avg[1]) * 1e5
svm_max = np.append(svm_arr[:, 2], svm_avg[2]) * 1e3
nn_max = np.append(np.array(tab["nn"])[:, 2], nn_avg[2]) * 1e3
ax[0].bar(x - w / 2, svm_rmse, w, label='SVM', color='#4C72B0')
ax[0].bar(x + w / 2, nn_rmse, w, label='MLP', color='#DD8452')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels + ['mean'])
ax[0].set_ylabel(r'RMSE ($\times10^{5}$)')
ax[0].set_title('RMSE')
ax[1].bar(x - w / 2, svm_max, w, label='SVM', color='#4C72B0')
ax[1].bar(x + w / 2, nn_max, w, label='MLP', color='#DD8452')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels + ['mean'])
ax[1].set_ylabel('max error ($\\times10^{-3}$)')
ax[1].set_title('Max error')
for a in ax:
    a.grid(alpha=0.3, axis='y')
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "cmp.png", dpi=200)
plt.close(fig)

# ---------------- 裁白边 ----------------
def crop(src, dst, pad=14):
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
               min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im = im.crop(box)
    im.save(dst)
    return dst

assets = {
    "data": crop(ROOT / "figs" / "冰型数据.png", FIG / "data.png"),
    "flow": crop(ROOT / "figs" / "流程图.png", FIG / "flow.png"),
    "contrast": crop(ROOT / "figs_contrast" / "svm_contrast.png", FIG / "contrast.png"),
    "pred": crop(ROOT / "figs_SVM" / "compare_SVM_258.png", FIG / "pred.png"),
    "energy": str(FIG / "energy.png"),
    "svm_err": str(FIG / "svm_err.png"),
    "cmp": str(FIG / "cmp.png"),
}

numbers = {
    "test_T": tab["test_T"],
    "lam": lam.tolist(), "ratio": ratio.tolist(), "cum": cum.tolist(),
    "recon_max": recon_max,
    "svm": tab["svm"], "nn": tab["nn"], "svm_avg": svm_avg.tolist(), "nn_avg": nn_avg.tolist(),
    "baseline_max": tab["baseline_max"], "baseline_rmse": tab["baseline_rmse"],
    "base_max_avg": base_max, "base_rmse_avg": base_rmse,
    "contrast": contrast,
    "svm_loo": svm_loo.tolist(), "nn_loo": nn_loo.tolist(),
    "t_pred_ms": t_pred,
    "assets": {k: str(v) for k, v in assets.items()},
}

(ROOT / "report_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")

print("POD 能量:", np.round(ratio, 4), " 累计:", np.round(cum, 4))
print("重构最大误差: %.3e" % recon_max)
print("SVM 平均:", np.round(svm_avg, 6))
print("MLP 平均:", np.round(nn_avg, 6))
print("基线 max/rmse:", round(base_max, 6), round(base_rmse, 6))
print("SVM LOO:", np.round(svm_loo, 6), " MLP LOO:", np.round(nn_loo, 6))
for r in contrast:
    print("C=%-5g eps=%-8g MAE=%.3e RMSE=%.3e Max=%.3e" % tuple(r))
print("单样本预测耗时: %.3f ms" % t_pred)
print("figs ->", FIG)
