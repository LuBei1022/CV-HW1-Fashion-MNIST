import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        self.x_cache = None
    
    def forward(self, x):
        z = x @ self.W + self.b
        self.x_cache = x
        return z

    def backward(self, dout):
        self.dW = self.x_cache.T @ dout
        self.db = np.sum(dout, axis = 0, keepdims = True)
        dx = dout @ self.W.T
        return dx

class ReLU:
    def __init__(self):
        self.x_cache = None
    
    def forward(self, x):
        self.x_cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        dx = np.where(self.x_cache > 0, dout, 0)
        return dx

    
class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None
    
    def forward(self, logits, y_true):
        self.y_true = y_true
        #求softmax
        max_logits = np.max(logits, axis = 1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis = 1, keepdims=True)

        batch_size = logits.shape[0]
        correct_probs = self.probs[np.arange(batch_size), y_true]
        #求loss
        loss = -np.mean(np.log(correct_probs + 1e-9))
        return loss
    
    def backward(self):
        dx = self.probs.copy()
        batch_size = self.y_true.shape[0]
        dx[np.arange(batch_size), self.y_true] -= 1
        dx = dx / batch_size
        return dx

class SimpleMLP:
    def __init__(self):
        self.fc1 = LinearLayer(in_features=28*28, out_features=128)
        self.relu = ReLU()
        self.fc2 = LinearLayer(in_features=128, out_features=10)
        self.loss_layer = SoftmaxCrossEntropy()
    
    def forward(self, x, y_true):
        fc1_forward = self.fc1.forward(x)
        relu_forward = self.relu.forward(fc1_forward)
        logits = self.fc2.forward(relu_forward)
        loss = self.loss_layer.forward(logits, y_true)
        return loss

    def backward(self):
        dout_loss = self.loss_layer.backward()
        dout_fc2 = self.fc2.backward(dout_loss)
        dout_relu = self.relu.backward(dout_fc2)
        dx = self.fc1.backward(dout_relu)
        return dx

class SGD:
    def __init__(self, learning_rate = 0.1):
        self.LR = learning_rate
    
    def step(self, model):
        #更新model的fc1层
        model.fc1.W = model.fc1.W - self.LR * model.fc1.dW
        model.fc1.b = model.fc1.b - self.LR * model.fc1.db
        #更新model的fc2层
        model.fc2.W = model.fc2.W - self.LR * model.fc2.dW
        model.fc2.b = model.fc2.b - self.LR * model.fc2.db

        



