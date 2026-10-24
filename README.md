# Perovskite Open-Data ML · 钙钛矿开源数据机器学习

三个围绕**光伏/钙钛矿公开数据集**的机器学习模块：表格回归、
序列决策（贝叶斯优化）、视觉检测。全部数据来自公开渠道，
仓库附下载脚本（内置国内镜像通道），每个模块独立可跑、结果图随仓库分发。

作者：ilo

| 模块 | 数据 | 核心结果 |
|---|---|---|
| [pce-regression](pce-regression/) | [Perovskite Database](https://www.perovskitedatabase.com) 4.2 万器件（CC-BY，84.7MB） | 工艺→PCE 回归 R²=0.657 / MAE=2.19%；混杂因子消融实验 |
| [bayesian-optimization](bayesian-optimization/) | 复用上者（虚拟实验室） | Optuna TPE 19.08±0.25% vs 随机 18.60±0.44%（3 种子）；最优解有真实样本支撑 |
| [defect-detection](defect-detection/) | [PV_Multi_Defect](https://huggingface.co/datasets/KeenForgeAI/PV_Multi_Defect-corrected)（HF，YOLO 格式） | YOLOv8 mAP50=0.762，5 类缺陷，混淆矩阵对角 0.82–0.99 |

## 主线

```
开源工艺数据库 ──► 回归模型（哪些工艺变量重要？）
                    │
                    ▼
        贝叶斯优化：固定实验预算下找最优工艺窗口
        （含混杂因子消融、多种子方差、可复现性工程）

器件质量另一只眼：YOLO 缺陷检测（数据→训练→mAP→部署全链）
```

比数字更值得看的是每个模块 README 里的**批判性分析**：数据质量混杂因子
如何扭曲特征重要性、BO 在廉价代理模型上为什么不占优、类别不均衡的代价。

## 快速开始

```bash
pip install -r requirements.txt

# 回归（先下数据，~85MB，内置断点续传与镜像方案）
cd pce-regression && python download_data.py && python process_pce_regression.py

# 贝叶斯优化（复用上一步数据）
cd ../bayesian-optimization && python virtual_lab_optuna.py

# 缺陷检测（~60MB 数据 + CPU 训练约 1.5 小时）
cd ../defect-detection && python download_data.py && python train_yolo.py
```

## License

代码 MIT；数据集许可见各模块说明（Perovskite Database: CC-BY 4.0）。
