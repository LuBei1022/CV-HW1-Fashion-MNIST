import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features) #hekaiming初始化
        self.b = np.zeros((1, out_features))
        self.x_cache = None
    
    def forward(self, x):
        z = x @ self.W + self.b
        self.x_cache = x
        return z

    def backward(self, dout, weight_decay = 0.0):
        self.dW = self.x_cache.T @ dout + weight_decay * self.W #L2正则化
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

class Sigmoid:
    def __init__(self):
        self.y_cache = None
    
    def forward(self,x):
        x_safe = np.clip(x, -500, 500)
        self.y_cache = 1.0 / (1.0 + np.exp(- x_safe))
        return self.y_cache
    
    def backward(self, dout):
        dx = dout * self.y_cache * (1 - self.y_cache)
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
    def __init__(self, input_dim = 784, hidden_dim1 = 128, hidden_dim2 = 56, output_dim = 10, activation1 = "ReLU", activation2 = "Sigmoid"):
        self.fc1 = LinearLayer(in_features = input_dim, out_features = hidden_dim1)
        if activation1 == "ReLU":
            self.act1 = ReLU()
        else:
            self.act1 = Sigmoid()
        self.fc2 = LinearLayer(in_features = hidden_dim1, out_features = hidden_dim2)
        if activation2 == "Sigmoid":
            self.act2 = Sigmoid()
        else:
            self.act2 = ReLU()
        self.fc3 = LinearLayer(in_features = hidden_dim2, out_features = output_dim)
        self.loss_layer = SoftmaxCrossEntropy()
    
    def forward(self, x, y_true):
        fc1_forward = self.fc1.forward(x)
        act1_forward = self.act1.forward(fc1_forward)
        fc2_forward = self.fc2.forward(act1_forward)
        act2_forward = self.act2.forward(fc2_forward)
        logits = self.fc3.forward(act2_forward)
        loss = self.loss_layer.forward(logits, y_true)
        return loss

    def backward(self, weight_decay = 0.0):
        dout_loss = self.loss_layer.backward()
        dout_fc3 = self.fc3.backward(dout_loss, weight_decay = weight_decay)
        dout_act2 = self.act2.backward(dout_fc3)
        dout_fc2 = self.fc2.backward(dout_act2, weight_decay = weight_decay)
        dout_act1 = self.act1.backward(dout_fc2)
        dx = self.fc1.backward(dout_act1, weight_decay = weight_decay)
        return dx
    
    def predict(self, x):
        fc1_forward = self.fc1.forward(x)
        act1_forward = self.act1.forward(fc1_forward)
        fc2_forward = self.fc2.forward(act1_forward)
        act2_forward = self.act2.forward(fc2_forward)
        logits = self.fc3.forward(act2_forward)
        return np.argmax(logits, axis=1) #选出logits里面得分最大的那个 作为predict的结果 返回的是最高分所在的索引即哪一类
    
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
        #更新model的f3层
        model.fc3.W = model.fc3.W - self.LR * model.fc3.dW
        model.fc3.b = model.fc3.b - self.LR * model.fc3.db




        



