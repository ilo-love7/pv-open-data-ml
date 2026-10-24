# -*- coding: utf-8 -*-
"""下载 PV_Multi_Defect-corrected 缺陷检测数据集（约 60MB，4184 个文件）到 data/。

来源：HuggingFace KeenForgeAI/PV_Multi_Defect-corrected（HF DOI 10.57967/hf/10882）
YOLO 格式：images/{train,val,test} + labels/{train,val,test} + data.yaml
运行：python download_data.py（国内走 hf-mirror 镜像，无需账号）
"""

# %%
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KeenForgeAI/PV_Multi_Defect-corrected",
    repo_type="dataset",
    local_dir=os.path.join("data", "PV_Multi_Defect"),
)
print("完成 -> data/PV_Multi_Defect/")
