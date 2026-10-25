# -*- coding: utf-8 -*-
"""Perovskite Database 原始字段的清洗工具（被回归与预测脚本共用）。"""

# %%
import re

import numpy as np
import pandas as pd


def last_num(x):
    """从 'Unknown >> 100.0' / '10 | 20' 这类多步值里取最后一个数值。

    数据库的工艺字段常记录多步操作（"退火1 >> 退火2"），此处约定取
    最后一步的数值；无法解析返回 NaN。
    """
    if pd.isna(x):
        return np.nan
    m = re.findall(r"[-+]?\d+\.?\d*", str(x))
    return float(m[-1]) if m else np.nan


def first_token(x):
    """多步类别值 'Spin-coating >> CBD' 取第一段，并归并 Unknown。

    归并成单一哨兵值 'Unknown/other'，便于后续作为独立类别处理——
    它往往代表"记录质量"，是重要的混杂信号，不能悄悄丢掉。
    """
    if pd.isna(x):
        return "Unknown/other"   # 缺失归入哨兵桶，而不是留给插补猜
    s = str(x).split(">>")[0].strip()
    return s if s and s.lower() != "unknown" else "Unknown/other"


def cap_categories(series, top_n=10):
    """高基数类别列：保留前 top_n 类，其余归 OTHER。返回 (新列, 保留类数)。"""
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), "OTHER"), len(top)
