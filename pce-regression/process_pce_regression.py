# -*- coding: utf-8 -*-
"""
项目9（开源版）：钙钛矿工艺参数 → 效率 回归建模
================================================
数据：The Perovskite Database Project (Jacobsson et al., Nat. Energy 2021)
      42,443 个器件记录 × 410 列，CC-BY 4.0 开源
      来源 GitHub: Jesperkemist/perovskitedatabase_data（Zenodo DOI 10.5281/zenodo.5834698）

任务：用工艺参数（沉积方式、淬灭介质、退火、气氛湿度、层结构、组分…）
      回归器件效率 JV_default_PCE，回答"哪些工艺变量最影响效率"。

流程：列筛选 → 清洗（数值提取/多步值合并/类别归并）→ 回归数据集
      → RF 与 HistGradientBoosting 两模型对比 → 特征重要性。
与实验室项目的衔接：samples.csv 里的工艺字段与本脚本特征一一对应，
      等 B-3 积累够样本后可直接套用同一套建模流程。
"""

# %%
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleaning import cap_categories, first_token, last_num
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = os.path.join("data", "perovskite_database_all.csv")
OUT = "."
os.makedirs(OUT, exist_ok=True)

# 目标列 + 特征列（数值 / 类别）
TARGET = "JV_default_PCE"
NUM_COLS = [
    "Cell_area_measured",
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "Perovskite_deposition_number_of_deposition_steps",
    "Perovskite_deposition_synthesis_atmosphere_relative_humidity",
    "JV_scan_speed",
    "Ref_publication_date",           # 只取年份
]
CAT_COLS = [
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_quenching_media",
    "Perovskite_deposition_synthesis_atmosphere",
    "Perovskite_deposition_thermal_annealing_atmosphere",
    "Perovskite_composition_short_form",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "Cell_architecture",
    "JV_test_atmosphere",
]
# 退火温度/时间形如 "Unknown >> 100.0"，单独清洗
ANNEAL_COLS = ["Perovskite_deposition_thermal_annealing_temperature",
               "Perovskite_deposition_thermal_annealing_time"]

USE_COLS = [TARGET] + NUM_COLS + CAT_COLS + ANNEAL_COLS + ["Module", "JV_measured"]


# %%
# ------------------------- 1. 读取与清洗 -------------------------


print("读取数据（只加载所需列）…")
df = pd.read_csv(DATA, usecols=USE_COLS, low_memory=False)
n0 = len(df)

# 只留单结点电池（排除组件/模块）、效率数值有效且在物理范围内
df = df[df["Module"] != True]                       # noqa: E712
pce = pd.to_numeric(df[TARGET], errors="coerce")
df = df.assign(y=pce)
df = df[(df["y"] > 0) & (df["y"] <= 30)]
n1 = len(df)

# 数值列清洗：所有数值列都可能有 "Unknown >> 100.0" / "10 | 20" 式多步值
for c in NUM_COLS + ANNEAL_COLS:
    if c == "Ref_publication_date":
        continue
    df[c] = df[c].map(last_num)
df["Ref_publication_date"] = df["Ref_publication_date"].astype(str).str[:4]
df["Ref_publication_date"] = pd.to_numeric(df["Ref_publication_date"], errors="coerce")

# 类别列清洗：取第一段 + 高基数归并（前10名之外归 OTHER）
cat_summary = []
for c in CAT_COLS:
    df[c] = df[c].map(first_token)
    df[c], k = cap_categories(df[c])
    cat_summary.append((c, k))

print(f"原始 {n0} 行 → 清洗后 {n1} 行（排除模块/无效PCE）")
print(f"PCE: 中位 {df['y'].median():.2f}%  均值 {df['y'].mean():.2f}%  最大 {df['y'].max():.2f}%")
for c, k in cat_summary:
    print(f"  [类别] {c}: 保留前{k}类+OTHER")

# %%
# ------------------------- 2. 建模 -------------------------
X = df[NUM_COLS + ANNEAL_COLS + CAT_COLS]
y = df["y"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS + ANNEAL_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), CAT_COLS),
])

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.1, random_state=0),
}

rows = []
fitted = {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5,
                         scoring="r2", n_jobs=-1)
    rows.append({"model": name, "R2_test": r2, "MAE_test": mae,
                 "R2_cv5_mean": cv.mean(), "R2_cv5_std": cv.std()})
    fitted[name] = pipe
    if name == "RandomForest":
        pred_rf = pred
    print(f"\n{name}: R²(test)={r2:.3f}  MAE={mae:.2f}%  "
          f"5折CV R²={cv.mean():.3f}±{cv.std():.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_metrics.csv"),
                          index=False, encoding="utf-8-sig")

# %%
# ------------------------- 3. 图 -------------------------
# 图1：PCE 分布
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["y"], bins=80, color="#5dade2", edgecolor="white")
ax.axvline(df["y"].median(), color="red", ls="--",
           label=f"中位 {df['y'].median():.2f}%")
ax.set_xlabel("JV_default_PCE (%)"); ax.set_ylabel("器件数")
ax.set_title(f"钙钛矿电池效率分布（清洗后 n={len(df):,}）")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_pce_hist.png"), dpi=150); plt.close(fig)

# 图2：预测 vs 实测（测试集）
best = max(rows, key=lambda r: r["R2_test"])["model"]
pred_te = fitted[best].predict(X_te)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_te, pred_te, s=4, alpha=0.3, color="#1f77b4")
lims = [0, 30]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("实测 PCE (%)"); ax.set_ylabel("预测 PCE (%)")
ax.set_title(f"{best} 测试集：R²={r2_score(y_te, pred_te):.3f}, "
             f"MAE={mean_absolute_error(y_te, pred_te):.2f}%")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_pred_vs_actual.png"), dpi=150); plt.close(fig)

# 图3：特征重要性（来自随机森林）
rf_pipe = fitted["RandomForest"]
feat_names = (NUM_COLS + ANNEAL_COLS +
              list(rf_pipe.named_steps["pre"]
                   .named_transformers_["cat"]
                   .named_steps["oh"].get_feature_names_out(CAT_COLS)))
imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                index=feat_names).sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index, imp.values, color="#48c9b0")
ax.set_xlabel("随机森林特征重要性")
ax.set_title("工艺特征对 PCE 的贡献 Top 15")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_feature_importance.png"), dpi=150)
plt.close(fig)
# %%
# ------------------------- 4. 混杂因子消融：剔除"淬灭介质"重训 -------------------------
# 原模型重要性第一名是 quenching_media 的 Unknown/other——数据质量混杂因子。
# 把它从特征里剔除重训：若排名大幅重排，说明模型确实在借"记录质量"刷相关性。
CAT_COLS2 = [c for c in CAT_COLS
             if c != "Perovskite_deposition_quenching_media"]
pre2 = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS + ANNEAL_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), CAT_COLS2),
])
lab2 = Pipeline([("pre", pre2), ("model", RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0))])
lab2.fit(X_tr, y_tr)
pred2 = lab2.predict(X_te)
print(f"[消融] 剔除淬灭介质后：R²={r2_score(y_te, pred2):.3f}  "
      f"MAE={mean_absolute_error(y_te, pred2):.2f}%  "
      f"(原模型 R²={r2_score(y_te, pred_rf):.3f})")

names2 = (NUM_COLS + ANNEAL_COLS +
          list(lab2.named_steps["pre"].named_transformers_["cat"]
               .named_steps["oh"].get_feature_names_out(CAT_COLS2)))
imp2 = pd.Series(lab2.named_steps["model"].feature_importances_, index=names2)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
for ax, (title, series) in zip(
        axes, [("含淬灭介质（原模型）", imp),
               ("剔除淬灭介质（消融）", imp2)]):
    top = series.sort_values(ascending=True).tail(8)
    ax.barh(top.index, top.values, color="#5499c7")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("随机森林特征重要性")
fig.suptitle("混杂因子消融：'淬灭介质 Unknown' 是否扭曲了重要性排名", fontsize=12)
fig.tight_layout()
fig.savefig("fig4_importance_ablation.png", dpi=150)
plt.close(fig)
print("消融对比图已保存 fig4_importance_ablation.png")
print("[消融] 新重要性 Top 5：")
for k, v in imp2.sort_values(ascending=False).head(5).items():
    print(f"  {k:55s} {v:.3f}")

# %%
# ------------------------- 5. 残差按组分分布：模型在哪类材料上失准 -------------------------
resid = pd.DataFrame({
    "composition": X_te["Perovskite_composition_short_form"].values,
    "resid_pct": (pred_rf - y_te.values),
})
groups = [g for _, g in resid.groupby("composition") if len(g) >= 30]
labels = [g["composition"].iloc[0] for g in groups]
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([g["resid_pct"] for g in groups], tick_labels=labels,
                showfliers=False, patch_artist=True)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
for b in bp["boxes"]:
    b.set_facecolor("#f5b7b1")
ax.axhline(0, color="red", ls="--", lw=1)
ax.set_ylabel("预测 - 实测 (PCE 百分点)")
ax.set_title("按组分分组的预测残差（仅显示 n>=30 的组，剔除离群点）")
fig.tight_layout()
fig.savefig("fig5_residual_by_composition.png", dpi=150)
plt.close(fig)
stats = resid.groupby("composition")["resid_pct"].agg(["count", "median"])
print("[残差分析] 按组分（中位残差，百分点）：")
print(stats[stats["count"] >= 30].sort_values("median").to_string())
print("残差分析图已保存 fig5_residual_by_composition.png")
