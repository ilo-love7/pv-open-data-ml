# -*- coding: utf-8 -*-
"""工艺参数 → 预测 PCE 命令行工具。

用法（全部参数可省略，省略时用贝叶斯优化找到的最优窗口）：
    python predict_pce.py                          # 最优窗口
    python predict_pce.py --T 110 --t 20 --thickness 400
    python predict_pce.py --T 150 --t 10 --humidity 0

首次运行自动训练模型并缓存到 model_rf.joblib（约 2 分钟）；
之后秒级响应。模型/缓存不入库（.gitignore 已排除）。
"""

# %%
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleaning import cap_categories, first_token, last_num  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "perovskite_database_all.csv")
MODEL = os.path.join(HERE, "model_rf.joblib")

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


# %%
def train_model(cache=MODEL):
    """训练并缓存随机森林；已有缓存则直接加载。"""
    import joblib
    if os.path.exists(cache):
        print("加载缓存模型", cache)
        return joblib.load(cache)

    if not os.path.exists(DATA):
        sys.exit(f"找不到数据 {DATA} —— 先运行 download_data.py")

    import re
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    def _last_num(x):
        if pd.isna(x):
            return np.nan
        m = re.findall(r"[-+]?\d+\.?\d*", str(x))
        return float(m[-1]) if m else np.nan

    df = pd.read_csv(DATA, low_memory=False,
                     usecols=[TARGET] + NUM_COLS + ANNEAL_COLS + CAT_COLS + ["Module"])
    df = df[df["Module"] != True]                          # noqa: E712
    df["y"] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[(df["y"] > 0) & (df["y"] <= 30)]
    for c in NUM_COLS + ANNEAL_COLS:
        if c != "Ref_publication_date":
            df[c] = df[c].map(_last_num)
    df["Ref_publication_date"] = pd.to_numeric(
        df["Ref_publication_date"].astype(str).str[:4], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].map(first_token)
        df[c], _ = cap_categories(df[c])

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS + ANNEAL_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), CAT_COLS)])
    from sklearn.ensemble import RandomForestRegressor
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=0))])
    pipe.fit(df[NUM_COLS + ANNEAL_COLS + CAT_COLS], df["y"])
    joblib.dump(pipe, cache)
    print("模型训练完成并缓存 ->", cache)
    return pipe


# %%
def predict(pipe, T=110, t=10, thickness=550, humidity=45,
            quench="Diethyl ether", composition="FAMAPbBrI",
            ETL="PCBM-60 | BCP", HTL="PTAA", back="Ag",
            year=2025):
    """固定测试条件与器件架构，只暴露可调工艺变量。"""
    row = pd.DataFrame([{
        "Cell_area_measured": 0.10,
        "JV_scan_speed": 100.0,
        "Ref_publication_date": year,
        "Perovskite_deposition_number_of_deposition_steps": 1,
        "Perovskite_deposition_procedure": "Spin-coating",
        "Perovskite_deposition_synthesis_atmosphere": "N2",
        "Perovskite_deposition_thermal_annealing_atmosphere": "N2",
        "Cell_architecture": "nip",
        "JV_test_atmosphere": "Air",
        "Perovskite_deposition_thermal_annealing_temperature": T,
        "Perovskite_deposition_thermal_annealing_time": t,
        "Perovskite_thickness": thickness,
        "Perovskite_deposition_synthesis_atmosphere_relative_humidity": humidity,
        "Perovskite_deposition_quenching_media": quench,
        "Perovskite_composition_short_form": composition,
        "ETL_stack_sequence": ETL,
        "HTL_stack_sequence": HTL,
        "Backcontact_stack_sequence": back,
        "Perovskite_band_gap": 1.58,
    }])
    return float(pipe.predict(row)[0])


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="工艺参数 → 预测 PCE（随机森林）")
    ap.add_argument("--T", type=float, default=110, help="退火温度 ℃")
    ap.add_argument("--t", type=float, default=10, help="退火时间 min")
    ap.add_argument("--thickness", type=float, default=550, help="钙钛矿厚度 nm")
    ap.add_argument("--humidity", type=float, default=45, help="合成气氛相对湿度 %%")
    ap.add_argument("--quench", default="Diethyl ether")
    ap.add_argument("--composition", default="FAMAPbBrI")
    ap.add_argument("--ETL", default="PCBM-60 | BCP")
    ap.add_argument("--HTL", default="PTAA")
    ap.add_argument("--back", default="Ag")
    args = ap.parse_args()

    pipe = train_model()
    pce = predict(pipe, **vars(args))
    print(f"\n预测 PCE = {pce:.2f}%")
    print("⚠️ 模型基于文献数据（R²≈0.66），绝对值仅供量级参考，相对比较更有意义。")
