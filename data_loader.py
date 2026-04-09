import numpy as np
from utils import mnist_reader

def load_fashion_mnist(data_path='data', val_size=10000):
    """
    加载并预处理 Fashion-MNIST 数据集
    """    
    # 1. 加载原始数据
    # data_path 默认指向当前目录下的 data 文件夹
    X_train_raw, y_train_raw = mnist_reader.load_mnist(data_path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')

    # 2. 数据预处理：归一化
    # 将像素值从 0-255 缩放到 0.0-1.0，并转换为 float32 类型，防止训练时梯度爆炸
    X_train_raw = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 显式展平 (Flatten)：确保数据一定是 (样本数, 784) 的二维矩阵
    # -1 的意思是让 NumPy 自动推算样本数量，28 * 28 明确展示了特征的来源
    X_train_raw = X_train_raw.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    
    # 3. 划分验证集 (Validation Set)
    # 为了保证类别分布均匀，必须先打乱数据
    np.random.seed(42) # 设置随机种子，保证每次运行打乱的顺序一样，方便后续找 Bug
    indices = np.arange(X_train_raw.shape[0])
    np.random.shuffle(indices)
    
    X_train_raw = X_train_raw[indices]
    y_train_raw = y_train_raw[indices]

    # 从 60000 个训练样本中，切分出前 val_size (默认10000) 个作为验证集
    X_val = X_train_raw[:val_size]
    y_val = y_train_raw[:val_size]
    
    # 剩下的 50000 个作为真正的训练集
    X_train = X_train_raw[val_size:]
    y_train = y_train_raw[val_size:]

    print(f"训练集形状: 图像 {X_train.shape}, 标签 {y_train.shape}")
    print(f"验证集形状: 图像 {X_val.shape}, 标签 {y_val.shape}")
    print(f"测试集形状: 图像 {X_test.shape},  标签 {y_test.shape}")

    # 返回三个元组，分别对应 训练集、验证集、测试集
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 测试代码：当你直接在终端运行 `python data_loader.py` 时，会执行下面这两行
if __name__ == "__main__":
    train_data, val_data, test_data = load_fashion_mnist(data_path='data')