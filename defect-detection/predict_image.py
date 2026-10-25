# -*- coding: utf-8 -*-
"""对光伏组件图像运行缺陷检测推理。

用法：
    python predict_image.py                          # 对 samples/ 内置样例推理
    python predict_image.py path/to/img1.jpg img2/   # 自定义图像或目录
    python predict_image.py --conf 0.5               # 调高置信度阈值

标注图输出到 results/infer/。模型权重随仓库分发（weights/best.pt，
mAP50=0.762，5 类缺陷），无需训练即可使用。
"""

# %%
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "best.pt")

CLASSES = ["black_border", "broken", "hot_spot", "no_electricity", "scratch"]


# %%
def main():
    ap = argparse.ArgumentParser(description="光伏组件缺陷检测推理")
    ap.add_argument("images", nargs="*",
                    help="图像文件或目录（缺省用 samples/ 内置样例）")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "infer"))
    args = ap.parse_args()

    if not os.path.exists(WEIGHTS):
        sys.exit(f"找不到权重 {WEIGHTS} —— 先跑 train_yolo.py 或从 release 下载")

    paths = args.images or [os.path.join(HERE, "samples", "sample_infer.jpg")]
    expanded = []
    for p in paths:
        if os.path.isdir(p):
            expanded += sorted(glob.glob(os.path.join(p, "*.jpg")))
        else:
            expanded.append(p)
    if not expanded:
        sys.exit("没有找到图像文件")

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    results = model.predict(expanded, conf=args.conf, save=True, project=args.out,
                            name="infer", exist_ok=True)

    print(f"\n{'图像':30s} 检出框")
    total = 0
    for r in results:
        names = r.boxes.cls.tolist()
        counts = {}
        for c in names:
            cls = CLASSES[int(c)]
            counts[cls] = counts.get(cls, 0) + 1
        total += len(names)
        summary = ", ".join(f"{k}×{v}" for k, v in counts.items()) or "无"
        print(f"{os.path.basename(r.path):30s} {len(names):3d} 个  [{summary}]")
    print(f"\n合计 {total} 个缺陷框；标注图已存 {args.out}/infer/")


if __name__ == "__main__":
    main()
