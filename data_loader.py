import numpy as np
from utils import mnist_reader

def load_fashion_mnist(data_path = "data"):
    #加载数据
    X_train_raw, y_train_raw = mnist_reader.load_mnist(path = data_path, kind = "train")
    X_test, y_test = mnist_reader.load_mnist(path = data_path, kind = "t10k")
    #数据归一化
    X_train_raw = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    #展平
    X_train_raw = X_train_raw.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    #打乱train_raw
    np.random.seed(42)
    index = np.arange(X_train_raw.shape[0])
    np.random.shuffle(index)
    X_train_raw = X_train_raw[index]
    y_train_raw = y_train_raw[index]
    #切分validation set
    X_validation = X_train_raw[:10000]
    y_validation = y_train_raw[:10000]
    X_train = X_train_raw[10000:]
    y_train = y_train_raw[10000:]
    return (X_train, y_train), (X_validation, y_validation),(X_test, y_test)

