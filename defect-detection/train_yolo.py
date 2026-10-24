# -*- coding: utf-8 -*-
"""
项目13（开源数据版）：光伏组件缺陷视觉检测（YOLOv8）
======================================================
数据：PV_Multi_Defect-corrected（KeenForge 整理版，HF DOI 10.57967/hf/10882）
      ~2000 张光伏组件图像 + YOLO 标注框，5 类缺陷：
      black_border(黑边) / broken(破损) / hot_spot(热斑) /
      no_electricity(断路) / scratch(划痕)
      HuggingFace 直接下载（走 hf-mirror），无需任何申请。

流程：数据校验 → 训练 yolov8n → 验证集 mAP → 测试图推理可视化。
说明：开源可得的缺陷图像是组件级红外/可见光图，不是钙钛矿薄膜显微图；
      但检测流水线（标注格式→训练→mAP→部署）与薄膜表面检测完全同构，
      将来把实验室拍的表面图按同样格式标注即可直接换数据。
"""

# %%
import os
import glob
import collections
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "PV_Multi_Defect")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["black_border", "broken", "hot_spot", "no_electricity", "scratch"]

# %%
# ------------------------- 1. 数据校验 -------------------------
stats = {}
for split in ("train", "val", "test"):
    imgs = glob.glob(os.path.join(DATA, "images", split, "*"))
    lbls = glob.glob(os.path.join(DATA, "labels", split, "*.txt"))
    n_box, cls_cnt = 0, collections.Counter()
    for lf in lbls:
        for ln in open(lf):
            p = ln.split()
            if len(p) == 5:
                n_box += 1
                cls_cnt[int(p[0])] += 1
    stats[split] = (len(imgs), len(lbls), n_box, cls_cnt)
    print(f"{split:5s}  图像 {len(imgs):5d}  标签 {len(lbls):5d}  框 {n_box:5d}  "
          f"类别分布 {dict(sorted(cls_cnt.items()))}")

# data.yaml 需要绝对路径
yaml_path = os.path.join(DATA, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
print("data.yaml 已更新为绝对路径")

# %%
# ------------------------- 2. 训练 yolov8n -------------------------
from ultralytics import YOLO

# 预训练权重：GitHub Releases 被墙时从 hf-mirror 拉（约 6MB）
if not os.path.exists("yolov8n.pt"):
    import urllib.request
    urllib.request.urlretrieve(
        "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
        "yolov8n.pt")
model = YOLO("yolov8n.pt")   # COCO 预训练权重，迁移学习
results = model.train(
    data=yaml_path,
    epochs=15,                # CPU 演示规模；正式用可加到 100+
    imgsz=640,
    batch=16,
    workers=4,
    project=os.path.join(ROOT, "runs"),
    name="pv_defect_yolov8n",
    exist_ok=True,
    patience=15,
)

# %%
# ------------------------- 3. 验证集指标 -------------------------
best_pt = os.path.join(ROOT, "runs", "pv_defect_yolov8n", "weights", "best.pt")
best = YOLO(best_pt)
metrics = best.val(data=yaml_path, split="val",
                   project=os.path.join(OUT), name="val_metrics", exist_ok=True)
print("\n========== 验证集结果 ==========")
print(f"mAP50     = {metrics.box.map50:.3f}")
print(f"mAP50-95  = {metrics.box.map:.3f}")
per_cls = metrics.box.maps  # 每类 mAP50-95
for c, m in zip(CLASSES, per_cls):
    print(f"  {c:16s} mAP50-95 = {m:.3f}")

# %%
# ------------------------- 4. 测试集推理可视化 -------------------------
test_imgs = sorted(glob.glob(os.path.join(DATA, "images", "test", "*")))[:6]
pred = best.predict(test_imgs, conf=0.4, save=True,
                    project=os.path.join(OUT), name="test_preds", exist_ok=True)
n_det = sum(len(r.boxes) for r in pred)
print(f"\n6 张测试图共检出 {n_det} 个缺陷框，标注图已存 results/test_preds/")

# 混淆矩阵图 ultralytics 已自动生成，拷贝一份到 results
cm = glob.glob(os.path.join(ROOT, "runs", "pv_defect_yolov8n", "*confusion_matrix*.png"))
for p in cm:
    shutil.copy(p, os.path.join(OUT, os.path.basename(p)))
pr = glob.glob(os.path.join(ROOT, "runs", "pv_defect_yolov8n", "results.png"))
for p in pr:
    shutil.copy(p, os.path.join(OUT, "training_curve.png"))
print("混淆矩阵与训练曲线已复制到 results/")
