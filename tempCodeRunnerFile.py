 best_combination = {"LR": LR, "hidden_dim1": hd1, "hidden_dim2": hd2, "weight_decay": weight_decay}
            weights = {
                'W1': model.fc1.W, 'b1': model.fc1.b,
                'W2': model.fc2.W, 'b2': model.fc2.b,
                'W3': model.fc3.W, 'b3': model.fc3.b
            }