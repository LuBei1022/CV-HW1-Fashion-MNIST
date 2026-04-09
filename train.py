import numpy as np
import os
import pickle  # 用于保存模型权重

# 导入我们之前写好的模块
from data_loader import load_fashion_mnist
from model import ThreeLayerMLP, SoftmaxCrossEntropyLoss

# ==========================================
# 辅助函数：计算准确率
# ==========================================
def calculate_accuracy(model, X, y, batch_size=256):
    """在给定数据集上计算准确率"""
    num_samples = X.shape[0]
    correct = 0
    
    # 为了防止一次性塞入太多数据导致内存爆炸，我们分批次测试
    for i in range(0, num_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        
        logits = model.forward(X_batch)
        # 找出预测概率最大的类别的索引
        predictions = np.argmax(logits, axis=1)
        correct += np.sum(predictions == y_batch)
        
    return correct / num_samples

# ==========================================
# 核心训练循环
# ==========================================
def train_model(X_train, y_train, X_val, y_val, 
                hidden_dim=128, 
                activation='relu',
                learning_rate=0.1, 
                lr_decay=0.95,
                weight_decay=1e-4, 
                batch_size=64, 
                epochs=20):
    """
    完整的训练引擎
    """
    print(f"\n--- 开始训练 [隐藏层:{hidden_dim}, 激活:{activation}, LR:{learning_rate}] ---")
    
    # 1. 初始化模型和损失函数
    model = ThreeLayerMLP(input_dim=784, hidden_dim=hidden_dim, num_classes=10, activation=activation)
    criterion = SoftmaxCrossEntropyLoss()
    
    num_samples = X_train.shape[0]
    best_val_acc = 0.0  # 记录历史最高验证集准确率
    history = {'train_loss': [], 'val_acc': []} # 用于最后画图
    
    # 如果没建过 weights 文件夹，就建一个
    if not os.path.exists('weights'):
        os.makedirs('weights')

    # 2. 开始 Epoch 循环（把所有训练数据看几遍）
    for epoch in range(epochs):
        # 每一轮开始前，先打乱训练数据（非常重要，防止模型背答案）
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        num_batches = num_samples // batch_size
        
        # 3. 开始 Batch 循环（每次看一小批图片）
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            # (1) 前向传播：计算预测值
            logits = model.forward(X_batch)
            
            # (2) 计算损失
            loss = criterion.forward(logits, y_batch)
            epoch_loss += loss
            
            # (3) 反向传播：计算梯度 (包含 L2 正则化)
            dlogits = criterion.backward()
            params_and_grads = model.backward(dlogits, weight_decay=weight_decay)
            
            # (4) SGD 优化器：更新权重 (W = W - learning_rate * dW)
            for param, grad in params_and_grads:
                param -= learning_rate * grad
                
        # 计算当前 Epoch 的平均 Loss
        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # 4. 学习率衰减 (Learning Rate Decay)
        # 随着训练进行，步子要迈小一点，这样能在谷底走得更稳
        learning_rate *= lr_decay
        
        # 5. 期中考试：在验证集上评估
        val_acc = calculate_accuracy(model, X_val, y_val)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {learning_rate:.4f}")
        
        # 6. 自动保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   [!] 发现新的最高分: {best_val_acc:.4f}，正在保存模型权重...")
            # 把模型的所有参数打包存成一个字典
            best_weights = {
                'W1': model.fc1.W, 'b1': model.fc1.b,
                'W2': model.fc2.W, 'b2': model.fc2.b,
                'W3': model.fc3.W, 'b3': model.fc3.b
            }
            # 序列化保存到硬盘
            with open('weights/best_model.pkl', 'wb') as f:
                pickle.dump(best_weights, f)

    print(f"\n训练结束！历史最高验证集准确率: {best_val_acc:.4f}")
    return history

# ==========================================
# 运行脚本的主入口
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fashion_mnist('data')
    
    # 2. 先用一组基础超参数跑一次看看效果
    # 真正的作业要求里，这里应该加上网格搜索（Grid Search）
    history = train_model(
        X_train, y_train, X_val, y_val,
        hidden_dim=128,
        activation='relu',
        learning_rate=0.1,    # 初始学习率
        lr_decay=0.95,        # 学习率每轮衰减系数
        weight_decay=1e-4,    # L2 正则化强度
        batch_size=64,        # 每次喂入 64 张图片
        epochs=10             # 先跑 10 轮测试一下
    )