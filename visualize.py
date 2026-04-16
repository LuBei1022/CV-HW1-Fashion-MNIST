import matplotlib.pyplot as plt
import numpy as np
import pickle

def visualize_first_layer_weights(W1, num_to_show=10):
    plt.figure(figsize=(15, 3))
    indices = np.linspace(0, W1.shape[1] - 1, num_to_show, dtype=int)
    
    for i, idx in enumerate(indices):
        # 提取第 idx 个神经元的权重 (784,)
        weight_vec = W1[:, idx]
        # 恢复成图像尺寸 28x28
        weight_img = weight_vec.reshape(28, 28)
        
        plt.subplot(1, num_to_show, i + 1)
        # 使用 'RdBu' 或 'gray' 颜色映射，RdBu 能看出正负权重
        plt.imshow(weight_img, cmap='RdBu')
        plt.title(f"Neuron {idx}")
        plt.axis('off')
    
    plt.suptitle("First Layer Weight Visualization (Spatial Patterns)")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_error_samples(model, X_test, y_test, class_names, max_samples=16):
    """
    精修版错例分析：准确提取错例，并用 4x4 网格展示
    """
    # 1. 预测并进行【准确率自检】
    predictions = model.predict(X_test)
    acc = np.mean(predictions == y_test)
    print(f"\n当前传入模型的准确率为: {acc * 100:.2f}%")
    
    # 2. 找到所有分错的索引
    error_mask = predictions != y_test
    error_indices = np.where(error_mask)[0]
    total_errors = len(error_indices)
    
    if total_errors == 0:
        print("模型完美！没有分错的样本。")
        return

    # 3. 按真实类别分组收集错例
    error_by_class = {i: [] for i in range(10)}
    for idx in error_indices:
        true_cls = y_test[idx]
        error_by_class[true_cls].append(idx)
        
    print(f"真实错例总数：{total_errors} / {len(y_test)}")

    # 4. 挑选展示样本（每个类别尽量挑一个，保证多样性）
    selected_indices = []
    for cls in range(10):
        if len(error_by_class[cls]) > 0:
            selected_indices.append(error_by_class[cls][0])
            
    # 如果不够 16 个，再按顺序补齐
    for idx in error_indices:
        if len(selected_indices) >= max_samples:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    # 5. 画出漂亮的 4x4 网格
    rows, cols = 4, 4
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = class_names[y_test[idx]]
        pred_label = class_names[predictions[idx]]
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        # 标题红字高亮，标明真假
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='darkred', fontsize=10)
        plt.axis('off')
        
    plt.suptitle(f"Error Analysis (Showing {len(selected_indices)} typical errors)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # 给主标题留点空
    plt.show()

if __name__ == "__main__":
    from data_loader import load_fashion_mnist
    _, _, (X_test, y_test) = load_fashion_mnist(data_path="data")
    
    # 读取权重
    with open('best_model.pkl', 'rb') as f:
        weights = pickle.load(f)
    
    from model import SimpleMLP
    model = SimpleMLP(input_dim=784, hidden_dim1=256, hidden_dim2=56, output_dim=10, 
                      activation1="ReLU", activation2="ReLU")
                      
    model.fc1.W, model.fc1.b = weights['W1'], weights['b1']
    model.fc2.W, model.fc2.b = weights['W2'], weights['b2']
    model.fc3.W, model.fc3.b = weights['W3'], weights['b3']
    
    print("\n正在生成第一层权重可视化图")
    visualize_first_layer_weights(model.fc1.W)

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    print("\n正在生成错例分析图...")
    plot_error_samples(model, X_test, y_test, class_names)