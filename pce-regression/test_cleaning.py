# -*- coding: utf-8 -*-
"""清洗工具单元测试：python -m pytest test_cleaning.py -v"""

# %%
import numpy as np
import pandas as pd
import pytest

from cleaning import cap_categories, first_token, last_num


# %%
def test_last_num_single():
    assert last_num("100.0") == 100.0
    assert last_num(100) == 100.0          # 数值输入直接转 float


def test_last_num_multistep_takes_last():
    assert last_num("Unknown >> 100.0") == 100.0
    assert last_num("10 | 20") == 20.0
    assert last_num("80 >> 100 >> 120.5") == 120.5


def test_last_num_unparseable():
    assert np.isnan(last_num("Unknown"))
    assert np.isnan(last_num(""))
    assert np.isnan(last_num(None))
    assert np.isnan(last_num(np.nan))


def test_first_token_takes_first_stage():
    assert first_token("Spin-coating >> CBD") == "Spin-coating"
    assert first_token("Blade-coating") == "Blade-coating"


def test_first_token_merges_unknown():
    assert first_token("Unknown") == "Unknown/other"
    assert first_token("Unknown >> Air") == "Unknown/other"   # 首段 Unknown 即哨兵
    assert first_token(None) == "Unknown/other"


def test_cap_categories():
    s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d", "e", "f"])
    out, k = cap_categories(s, top_n=3)
    assert k == 3
    assert (out.iloc[:5] == "a").all()
    assert (out.iloc[10:] == "OTHER").all()  # c(8-9) 之后全归 OTHER
    assert not out.isin(["d", "e", "f"]).any()
