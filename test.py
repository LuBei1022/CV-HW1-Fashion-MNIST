import numpy as np
import pickle

from data_loader import load_fashion_mnist
from model import SimpleMLP

#混淆矩阵
def confusion_matrix(y_true, y_predict, num_class = 10):
    cm = np.zeros((num_class, num_class), dtype = int)
    for i, j in zip(y_true, y_predict):
        cm[i, j] += 1
    return cm

if __name__ == "__main__":
    _, _, (X_test, y_test) = load_fashion_mnist(data_path="data")
    model = SimpleMLP(input_dim=784, hidden_dim1=256, hidden_dim2=56, output_dim=10, 
                      activation1="ReLU", activation2="ReLU") #用的是search里面得到best combination
    with open("best_model.pkl", "rb") as f:
        best_weight = pickle.load(f)
    
    model.fc1.W, model.fc1.b = best_weight['W1'], best_weight['b1']
    model.fc2.W, model.fc2.b = best_weight['W2'], best_weight['b2']
    model.fc3.W, model.fc3.b = best_weight['W3'], best_weight['b3']
    #在test集里面测试
    prediction = model.predict(X_test)
    test_accuracy = np.mean(prediction == y_test)
    print(f"测试集的准确率:{test_accuracy}\n")
    #创建混淆矩阵
    cm = confusion_matrix(y_true= y_test, y_predict = prediction, num_class=10)
    class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] #来源是GitHub mnist

    print("混淆矩阵：\n")
    header = f"{'True / Pred':>12}" + "".join([f"{i:>6}" for i in range(10)])
    print(header)

    for i in range(10):
        row_str = "".join([f"{val:>6d}" for val in cm[i]])
        print(f"{class_name[i]:>12} | {row_str}")