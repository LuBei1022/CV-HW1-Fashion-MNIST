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

## 运行训练
本项目按照“超参数搜索 -> 正式训练 -> 模型评估”的逻辑设计。

### 0. 数据处理

数据存放：请确保 data/ 文件夹下包含解压后的 Fashion-MNIST 原始字节文件。

### 1. 超参数搜索

```bash
python search.py
```

功能：自动遍历学习率、隐藏层维度和正则化强度组合。

输出：终端将实时打印每组参数的验证集准确率，并记录全局最优组合。

### 2. 模型训练
使用搜索得到的最优参数（已在代码中预设）启动正式训练：

```bash
python train.py
```

每轮训练后自动在验证集上评估。

自动保存：当验证集准确率刷新记录时，脚本会自动将权重持久化为 best_model.pkl。

可视化：训练结束后自动保存 learning_curves.png。

### 3. 测试结果与可视化
训练完成后（确保目录下已有 best_model.pkl），运行评估脚本：

```bash
python test.py
```

产出结果：

最终精度：终端输出模型在 10000 张独立测试集上的准确率。

混淆矩阵：打印各类别的详细误判分布。

权重特征图：生成 weight_visualization.png，展示第一层神经元学到的空间模式。

错例分析：生成 error_analysis.png，直观展示 16 张典型分类错误的图像。