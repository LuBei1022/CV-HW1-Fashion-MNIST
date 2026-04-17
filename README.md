# CV-HW1-Fashion-MNIST

这是一个完全基于 NumPy 的 MLP 项目，用于解决 Fashion-MNIST 数据集的图像分类任务。该项目不依赖 PyTorch、TensorFlow 等任何深度学习框架，核心逻辑（包括前向传播、反向传播、权重更新等）均采用底层矩阵运算实现。
## 环境依赖
本项目运行环境建议使用 Python 3.8+。
```bash
pip install numpy matplotlib
numpy>=1.20.0
matplotlib>=3.3.0
pickle
```

## 项目结构
```text

├── data/               # 存放 Fashion-MNIST 原始数据
├── best_model.pkl      # 训练完成的最佳模型权重文件
├── data_loader.py      # 数据加载、预处理、归一化与划分逻辑
├── model.py            # MLP 核心架构与各层实现 (Linear, ReLU, Sigmoid等)
├── search.py           # 超参数网格搜索脚本
├── train.py            # 正式训练脚本（带模型保存机制）
├── test.py             # 测试集评估、权重可视化与错例分析脚本
└── README.md           # 项目说明文档
```
