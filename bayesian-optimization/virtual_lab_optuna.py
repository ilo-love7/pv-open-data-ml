# -*- coding: utf-8 -*-
"""
项目12：贝叶斯优化自动寻优工艺窗口（self-driving lab 入门）
============================================================
思路：用项目9的随机森林（工艺→PCE 回归，R²=0.657）作为"虚拟实验室"，
在真实工艺参数范围内做贝叶斯优化（Optuna TPE），与随机搜索对比，
找出预测效率最高的工艺窗口。

关键设计（对应项目9发现的两个混杂因子）：
  1. 淬灭介质只在【真实类别】里搜索，排除 "Unknown/other"
     ——否则模型会钻数据质量的空子（Unknown 特征重要性第一名的教训）
  2. 组分也只在真实的前几类里选，band_gap 按所选组分的训练集中位数
     固定（band gap 由组分决定，不该独立乱搜）
  3. 测试条件（面积/扫描速度/年份）固定为典型值，不给模型钻空子的自由度

产出：
  results/fig1_optimization_trace.png —— TPE vs 随机搜索收敛曲线
  results/fig2_param_importance.png   —— fANOVA 参数重要性
  results/fig3_best_window.png        —— 最优解在退火温度×时间平面的位置
  results/best_params.json            —— 最优工艺窗口
"""

# %%
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = os.path.join("..", "pce-regression", "data", "perovskite_database_all.csv")
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# 与项目9保持一致的特征体系
TARGET = "JV_default_PCE"
NUM_COLS = ["Cell_area_measured", "Perovskite_thickness", "Perovskite_band_gap",
            "Perovskite_deposition_number_of_deposition_steps",
            "Perovskite_deposition_synthesis_atmosphere_relative_humidity",
            "JV_scan_speed", "Ref_publication_date"]
ANNEAL_COLS = ["Perovskite_deposition_thermal_annealing_temperature",
               "Perovskite_deposition_thermal_annealing_time"]
CAT_COLS = ["Perovskite_deposition_procedure",
            "Perovskite_deposition_quenching_media",
            "Perovskite_deposition_synthesis_atmosphere",
            "Perovskite_deposition_thermal_annealing_atmosphere",
            "Perovskite_composition_short_form",
            "ETL_stack_sequence", "HTL_stack_sequence",
            "Backcontact_stack_sequence", "Cell_architecture",
            "JV_test_atmosphere"]


def last_num(x):
    if pd.isna(x):
        return np.nan
    m = re.findall(r"[-+]?\d+\.?\d*", str(x))
    return float(m[-1]) if m else np.nan


def first_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).split(">>")[0].strip()
    return s if s and s.lower() != "unknown" else "Unknown/other"


# %%
# ------------------------- 1. 数据清洗 + 训练虚拟实验室模型 -------------------------
USE = [TARGET] + NUM_COLS + ANNEAL_COLS + CAT_COLS + ["Module"]
df = pd.read_csv(DATA, usecols=USE, low_memory=False)
df = df[df["Module"] != True]                                  # noqa: E712
df["y"] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[(df["y"] > 0) & (df["y"] <= 30)]

for c in NUM_COLS + ANNEAL_COLS:
    if c == "Ref_publication_date":
        continue
    df[c] = df[c].map(last_num)
df["Ref_publication_date"] = pd.to_numeric(
    df["Ref_publication_date"].astype(str).str[:4], errors="coerce")
for c in CAT_COLS:
    df[c] = df[c].map(first_token)
    top = df[c].value_counts().head(10).index
    df[c] = df[c].where(df[c].isin(top), "OTHER")

X, y = df[NUM_COLS + ANNEAL_COLS + CAT_COLS], df["y"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS + ANNEAL_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), CAT_COLS)])
lab = Pipeline([("pre", pre), ("model", RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=0))])  # 单线程预测：并行浮点顺序噪声会被TPE采样放大，破坏可复现性
lab.fit(X_tr, y_tr)
print(f"虚拟实验室就绪：RF 测试集 R²={r2_score(y_te, lab.predict(X_te)):.3f}")

# 各组分对应的典型带隙（训练集中位数），搜索时按组分锁定
gap_by_comp = df.groupby("Perovskite_composition_short_form")["Perovskite_band_gap"] \
                .median().to_dict()
# 训练集数值列（供支撑度检查）
num_tr = X_tr[NUM_COLS + ANNEAL_COLS]


# %%
# ------------------------- 2. 搜索空间与目标函数 -------------------------
# 只在真实类别中搜索（排除 Unknown/other 和 OTHER —— 项目9混杂因子教训）
REAL_QUENCH = [c for c in df["Perovskite_deposition_quenching_media"].value_counts().head(5).index
               if c not in ("Unknown/other", "OTHER")]
REAL_COMP = [c for c in df["Perovskite_composition_short_form"].value_counts().head(4).index
             if c != "OTHER"]
REAL_ETF = [c for c in df["ETL_stack_sequence"].value_counts().head(5).index
            if c != "OTHER"]
REAL_HTL = [c for c in df["HTL_stack_sequence"].value_counts().head(5).index
            if c != "OTHER"]
REAL_BACK = [c for c in df["Backcontact_stack_sequence"].value_counts().head(3).index
             if c != "OTHER"]
print("淬灭介质候选:", REAL_QUENCH)
print("组分候选:", REAL_COMP)


def suggest_params(trial):
    p = {
        # 可控连续工艺变量（真实范围）
        "annealing_T": trial.suggest_float("annealing_T", 80, 150, step=5),      # ℃
        "annealing_t": trial.suggest_float("annealing_t", 5, 120, step=5),       # min
        "thickness": trial.suggest_float("thickness", 150, 800, step=25),        # nm
        "humidity": trial.suggest_float("humidity", 0, 60, step=5),              # %
        # 真实离散候选
        "quench": trial.suggest_categorical("quench", REAL_QUENCH),
        "composition": trial.suggest_categorical("composition", REAL_COMP),
        "ETL": trial.suggest_categorical("ETL", REAL_ETF),
        "HTL": trial.suggest_categorical("HTL", REAL_HTL),
        "back": trial.suggest_categorical("back", REAL_BACK),
    }
    return p


def build_row(p):
    """把搜索参数组装成模型输入行；其余变量固定为典型值。"""
    row = {
        "Cell_area_measured": 0.10,
        "JV_scan_speed": df["JV_scan_speed"].median(),
        "Ref_publication_date": 2025,
        "Perovskite_deposition_number_of_deposition_steps": 1,
        "Perovskite_deposition_procedure": "Spin-coating",
        "Perovskite_deposition_synthesis_atmosphere": "N2",
        "Perovskite_deposition_thermal_annealing_atmosphere": "N2",
        "Cell_architecture": "nip",
        "JV_test_atmosphere": "Air",
        "Perovskite_deposition_thermal_annealing_temperature": p["annealing_T"],
        "Perovskite_deposition_thermal_annealing_time": p["annealing_t"],
        "Perovskite_thickness": p["thickness"],
        "Perovskite_deposition_synthesis_atmosphere_relative_humidity": p["humidity"],
        "Perovskite_deposition_quenching_media": p["quench"],
        "Perovskite_composition_short_form": p["composition"],
        "ETL_stack_sequence": p["ETL"],
        "HTL_stack_sequence": p["HTL"],
        "Backcontact_stack_sequence": p["back"],
        # 带隙由组分锁定（中位数），防止模型把"带隙-组分"组合推出物理范围
        "Perovskite_band_gap": gap_by_comp.get(p["composition"], 1.58),
    }
    return pd.DataFrame([row])


def objective(trial):
    p = suggest_params(trial)
    pred = float(lab.predict(build_row(p))[0])
    trial.set_user_attr("pred", pred)
    return pred


# %%
# ------------------------- 3. TPE 贝叶斯优化 vs 随机搜索 -------------------------
N_TRIALS = 150
SEEDS = [42, 43, 44]          # 多种子：单一随机种子下的"侥幸命中"不可信
studies = {}                  # name -> 第一个种子的 study（画轨迹用）
finals = {}                   # name -> 各种子的最优值列表
for name, sampler_cls in (("TPE贝叶斯优化", optuna.samplers.TPESampler),
                          ("随机搜索", optuna.samplers.RandomSampler)):
    finals[name] = []
    for seed in SEEDS:
        st = optuna.create_study(direction="maximize",
                                 sampler=sampler_cls(seed=seed))
        st.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        finals[name].append(st.best_value)
        if seed == SEEDS[0]:
            studies[name] = st
    print(f"{name}: 各种子最优={['%.2f' % v for v in finals[name]]}  "
          f"均值 {np.mean(finals[name]):.2f} +/- {np.std(finals[name]):.2f}%")

# %%
# ------------------------- 4. 图与结果输出 -------------------------
# 图1：收敛曲线（历史最优 vs 试验次数）
fig, ax = plt.subplots(figsize=(7.5, 5))
for name, style in (("TPE贝叶斯优化", "-"), ("随机搜索", "--")):
    vals = [t.value for t in studies[name].trials]
    best_so_far = pd.Series(vals).cummax()
    ax.plot(range(1, N_TRIALS + 1), best_so_far, style, lw=2,
            label=f"{name}（3种子最优 {np.mean(finals[name]):.2f}"
                  f"+/-{np.std(finals[name]):.2f}%）")
ax.set_xlabel("试验次数（虚拟实验次数）")
ax.set_ylabel("历史最优预测 PCE (%)")
ax.set_title("TPE vs 随机搜索（各150次虚拟实验，3个随机种子）：\n"
             "平滑代理模型上两者统计不可分（见图例均值+/-标准差）——BO 的收益要靠昂贵实验才体现")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_optimization_trace.png"), dpi=150)
plt.close(fig)

# 图2：fANOVA 参数重要性
best_st = studies["TPE贝叶斯优化"]
imp = optuna.importance.get_param_importances(best_st)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ks = list(imp.keys())[::-1]
ax.barh(ks, [imp[k] for k in imp.keys()][::-1], color="#af7ac5")
ax.set_xlabel("fANOVA 重要性")
ax.set_title("哪些搜索变量最影响预测 PCE（基于 TPE 全部试验）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_param_importance.png"), dpi=150)
plt.close(fig)

# 图3：最优解在 退火温度×时间 平面 + 训练样本密度
bp = best_st.best_params
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(num_tr["Perovskite_deposition_thermal_annealing_temperature"],
                num_tr["Perovskite_deposition_thermal_annealing_time"],
                s=4, alpha=0.15, color="#bdc3c7", label="训练样本（退火温度×时间）")
ax.scatter(bp["annealing_T"], bp["annealing_t"], marker="*", s=600,
           color="#c0392b", zorder=5, label=f"TPE 最优 ({best_st.best_value:.1f}%)")
ax.set_xlabel("退火温度 (℃)"); ax.set_ylabel("退火时间 (min)")
ax.set_title("最优工艺窗口位置（灰点=训练数据分布，星=最优解）")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_best_window.png"), dpi=150)
plt.close(fig)

# 最优解支撑度检查：附近有没有真实训练样本（防止钻进数据空隙）
near = ((np.abs(num_tr["Perovskite_deposition_thermal_annealing_temperature"] - bp["annealing_T"]) <= 10)
        & (np.abs(num_tr["Perovskite_deposition_thermal_annealing_time"] - bp["annealing_t"]) <= 10)
        & (np.abs(num_tr["Perovskite_thickness"] - bp["thickness"]) <= 50))
n_near = int(near.sum())

best_full = {**bp, "predicted_PCE": round(best_st.best_value, 2),
             "tpe_best_3seeds": [round(v, 2) for v in finals["TPE贝叶斯优化"]],
             "rand_best_3seeds": [round(v, 2) for v in finals["随机搜索"]],
             "n_train_samples_nearby": n_near}
with open(os.path.join(OUT, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(best_full, f, ensure_ascii=False, indent=2)

print("\n========== 最优工艺窗口 ==========")
for k, v in best_full.items():
    print(f"  {k}: {v}")
print(f"\n⚠️ 绝对值是代理模型外推，仅供参考量级与相对排序；")
print(f"   附近 {n_near} 个真实训练样本支撑该区域（<10 即需谨慎）。")
print("图与 best_params.json 已保存到 results/")
