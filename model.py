import numpy as np

#Linear Layer
class LinearLayer:
        def __init__(self, in_features, out_features):
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)  # He 初始化，适用于 ReLU 激活函数
            self.b = np.zeros((1, out_features))
            self.x_cache = None
        
        def forward(self, x):
            self.x_cache = x 
            output = self.x_cache @ self.W +self.b
            return output
        
        def backward(self, dout):
            #根据链式法则计算梯度：dW = x^T * dout, db = sum(dout), dx = dout * W^T
            self.dW = self.x_cache.T @ dout
            self.db = np.sum(dout,axis = 0, keepdims = True)
            dx = dout @ self.W.T
            return dx

#激活函数
class ReLU:
    def __init__(self):
        self.x_cache = None
    
    def forward(self, x):
        self.x_cache = x
        out = np.maximum(0, x)  # ReLU 激活函数：输出为输入和0的较大值
        return out

    def backward(self, dout):
        x = self.x_cache
        dx = dout * (x > 0)  # ReLU 激活函数的梯度：输入大于0时梯度为1，否则为0
        return dx

class Sigmoid:
    def __init__(self):
        self.out_cache = None
    
