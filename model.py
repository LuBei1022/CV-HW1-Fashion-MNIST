import numpy as np

# ==========================================
# 1. 基础网络层定义 (Linear Layer)
# ==========================================
class LinearLayer:
    def __init__(self, in_features, out_features):
        # 使用 He 初始化 (适合 ReLU)，防止初始梯度消失或爆炸
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        
        # 缓存前向传播的输入，反向传播计算梯度时要用
        self.x_cache = None 
        # 权重的梯度缓存
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_cache = x
        # Z = X * W + b
        return np.dot(x, self.W) + self.b

    def backward(self, dout, weight_decay=0.0):
        """
        dout: 上一层传下来的梯度 (dL/dZ)
        weight_decay: L2 正则化系数
        """
        # 计算权重和偏置的梯度 (考虑 L2 正则化项)
        self.dW = np.dot(self.x_cache.T, dout) + weight_decay * self.W
        self.db = np.sum(dout, axis=0, keepdims=True)
        
        # 计算传递给下一层的梯度 (dL/dX)
        dx = np.dot(dout, self.W.T)
        return dx

# ==========================================
# 2. 激活函数定义 (Activations)
# ==========================================
class ReLU:
    def __init__(self):
        self.x_cache = None

    def forward(self, x):
        self.x_cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        # ReLU 的导数：x <= 0 时梯度为 0，x > 0 时梯度保持不变
        dx[self.x_cache <= 0] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.out_cache = None

    def forward(self, x):
        # 防止 exp 溢出的小技巧
        x_safe = np.clip(x, -500, 500)
        self.out_cache = 1.0 / (1.0 + np.exp(-x_safe))
        return self.out_cache

    def backward(self, dout):
        # Sigmoid 导数: f(x) * (1 - f(x))
        return dout * self.out_cache * (1.0 - self.out_cache)

# ==========================================
# 3. 损失函数 (Softmax + Cross Entropy)
# ==========================================
class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: 网络的最后输出 (未经过 softmax)
        y_true: 真实标签，形如 [0, 2, 1, ...]
        """
        self.y_true = y_true
        batch_size = logits.shape[0]
        
        # 减去最大值防止指数爆炸 (数值稳定性优化)
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits_shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 计算交叉熵 Loss
        correct_logprobs = -np.log(self.probs[np.arange(batch_size), y_true] + 1e-9)
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def backward(self):
        """
        Softmax 和 交叉熵 结合求导的结果非常优雅：P - Y
        """
        batch_size = self.probs.shape[0]
        dlogits = self.probs.copy()
        # 对应正确类别的概率减去 1
        dlogits[np.arange(batch_size), self.y_true] -= 1.0
        dlogits /= batch_size
        return dlogits

# ==========================================
# 4. 组装三层神经网络 (MLP)
# ==========================================
class ThreeLayerMLP:
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, activation='relu'):
        """
        包含两个隐藏层和一个输出层，共三个权重矩阵。
        """
        self.fc1 = LinearLayer(input_dim, hidden_dim)
        self.fc2 = LinearLayer(hidden_dim, hidden_dim)
        self.fc3 = LinearLayer(hidden_dim, num_classes)
        
        # 根据字符串选择激活函数
        if activation.lower() == 'relu':
            self.act1 = ReLU()
            self.act2 = ReLU()
        elif activation.lower() == 'sigmoid':
            self.act1 = Sigmoid()
            self.act2 = Sigmoid()
        else:
            raise ValueError("只支持 relu 或 sigmoid")

    def forward(self, x):
        # 第一层
        out1 = self.fc1.forward(x)
        act1_out = self.act1.forward(out1)
        # 第二层
        out2 = self.fc2.forward(act1_out)
        act2_out = self.act2.forward(out2)
        # 输出层 (分类任务不需要在最后一层加激活函数，直接输出 logits)
        logits = self.fc3.forward(act2_out)
        return logits

    def backward(self, dlogits, weight_decay=0.0):
        # 链式法则：从后往前推导
        dout2 = self.fc3.backward(dlogits, weight_decay)
        dact2 = self.act2.backward(dout2)
        
        dout1 = self.fc2.backward(dact2, weight_decay)
        dact1 = self.act1.backward(dout1)
        
        _ = self.fc1.backward(dact1, weight_decay)
        
        # 返回所有需要更新的参数及其梯度
        params_and_grads = [
            (self.fc1.W, self.fc1.dW), (self.fc1.b, self.fc1.db),
            (self.fc2.W, self.fc2.dW), (self.fc2.b, self.fc2.db),
            (self.fc3.W, self.fc3.dW), (self.fc3.b, self.fc3.db)
        ]
        return params_and_grads

# 测试代码
if __name__ == "__main__":
    # 模拟一个 batch 的数据: 32 张图片
    dummy_x = np.random.randn(32, 784)
    dummy_y = np.random.randint(0, 10, size=(32,))

    model = ThreeLayerMLP(activation='relu')
    criterion = SoftmaxCrossEntropyLoss()

    # 前向传播
    logits = model.forward(dummy_x)
    loss = criterion.forward(logits, dummy_y)
    
    # 反向传播
    dlogits = criterion.backward()
    grads = model.backward(dlogits, weight_decay=1e-4)

    print(f"模型前向传播成功，初始 Loss: {loss:.4f}")
    print(f"反向传播成功，共生成 {len(grads)} 组参数梯度")