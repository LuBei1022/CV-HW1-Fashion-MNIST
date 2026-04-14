import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from data_loader import load_fashion_mnist
from model import SimpleMLP, SGD 

def calculate_accuracy(model, X, y):
    prediction = model.predict(X)
    comparison = (prediction == y)
    accuracy = np.mean(comparison)
    return accuracy

def plot_learning_curves(history):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 画 Loss 曲线
    ax1.plot(history['training_loss'], label='Train Loss', color='tab:red', marker='o')
    ax1.set_title('Training Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # 画 Accuracy 曲线
    ax2.plot(history['validation_accuracy'], label='Validation Accuracy', color='tab:blue', marker='s')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_validation, y_validation),(X_test, y_test) = load_fashion_mnist(data_path = "data")
    model = SimpleMLP(input_dim = 784, hidden_dim1 = 128, hidden_dim2 = 56, output_dim = 10, activation1 = "ReLU", activation2 = "Sigmoid")
    optimizer = SGD(learning_rate = 0.1)
    #设置超参数
    epochs = 20
    batch_size = 64
    weight_decay = 0.001
    lr_decay = 0.95
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    history = {'training_loss': [], 'validation_accuracy': []}
    best_validation_accuracy = 0.0

    print("开始训练\n")
    for epoch in range(epochs):
        #打乱
        index = np.random.permutation(num_samples)
        X_train_shuffled = X_train[index]
        y_train_shuffled = y_train[index]
        epoch_loss = 0.0

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            #送进去跑
            loss = model.forward(X_train_shuffled[start_index: end_index], y_train_shuffled[start_index: end_index])
            model.backward(weight_decay=weight_decay) #目的是更新3个dW
            optimizer.step(model)
            epoch_loss += loss
        #一轮训练结束后的操作
        avg_train_loss = epoch_loss / num_batches
        history["training_loss"].append(avg_train_loss)
        #在验证集上面测试
        validation_accuracy = calculate_accuracy(model, X_validation, y_validation)
        history["validation_accuracy"].append(validation_accuracy)

        optimizer.LR = optimizer.LR * lr_decay
        print(f"第 {epoch+1:02d}/{epochs} 轮 | Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {validation_accuracy * 100:.2f}% | 当前 LR: {optimizer.LR:.4f}")
        
        #保存最好的
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            weights = {
                'W1': model.fc1.W, 'b1': model.fc1.b,
                'W2': model.fc2.W, 'b2': model.fc2.b,
                'W3': model.fc3.W, 'b3': model.fc3.b
            }
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(weights, f)

    plot_learning_curves(history)