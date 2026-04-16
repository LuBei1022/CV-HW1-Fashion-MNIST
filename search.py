import numpy as np
import pickle
import itertools

from data_loader import load_fashion_mnist
from model import SimpleMLP, SGD
from train import calculate_accuracy

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), _ = load_fashion_mnist(data_path="data")
    #参数初始化
    num_samples = X_train.shape[0]
    batch_size = 64
    num_batches = num_samples // batch_size
    epochs = 30 
    lr_decay = 0.95
    #采用grid serch
    LR = [0.15, 0.1, 0.05, 0.01]
    hd1 = [128, 256]
    hd2 = [56, 64, 128]
    weight_decay = [0.0001, 0.001, 0.01]
    #组合起来
    combination = list(itertools.product(LR, hd1,hd2, weight_decay))
    global_best_accuracy = 0.0
    best_combination = {}
    #开始寻找
    for idx, (LR, hd1, hd2, weight_decay) in enumerate(combination):
        #造model
        model = SimpleMLP(input_dim = 784, hidden_dim1 = hd1, hidden_dim2=hd2, output_dim=10, activation1="ReLU", activation2="ReLU")
        optimizer = SGD(learning_rate=LR)
        current_best_accuracy = 0.0
        #早停法的初始化
        patience = 3         
        patience_counter = 0
        #开训
        for epoch in range(epochs):
            #打乱
            index = np.random.permutation(num_samples)
            X_train_shuffled = X_train[index]
            y_train_shuffled = y_train[index]
            #batch训练
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = start_index + batch_size
                loss = model.forward(X_train_shuffled[start_index:end_index], y_train_shuffled[start_index:end_index])
                model.backward(weight_decay=weight_decay) 
                optimizer.step(model)
            #验证集
            val_acc = calculate_accuracy(model, X_val, y_val)
            optimizer.LR *= lr_decay
            print(f"Epoch {epoch + 1} validation accuracy: {val_acc * 100:.2f}%")

            if val_acc > current_best_accuracy:
                current_best_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > patience:
                print("早停，本轮暂停\n")
                break
            
        if current_best_accuracy > global_best_accuracy:
            global_best_accuracy = current_best_accuracy
            best_combination = {"LR": LR, "hidden_dim1": hd1, "hidden_dim2": hd2, "weight_decay": weight_decay}
            weights = {
                'W1': model.fc1.W, 'b1': model.fc1.b,
                'W2': model.fc2.W, 'b2': model.fc2.b,
                'W3': model.fc3.W, 'b3': model.fc3.b
            }
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(weights, f)
    
    print(f"global best accuracy:{global_best_accuracy}\n")
    print(f"best combination:{best_combination}\n")


        



