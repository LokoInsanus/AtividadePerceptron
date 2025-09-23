import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

def train_with_early_stopping(X, y, patience=5, n_epochs=100, learning_rate=0.1):
    # Divide treino/validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(learning_rate=learning_rate, n_epochs=1)  # treinaremos 1 época por vez

    # Inicializar pesos manualmente
    n_features = X_train.shape[1]
    perceptron.weights = np.zeros(n_features)
    perceptron.bias = 0

    best_val_acc = -np.inf
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # Uma época de treino
        errors = 0
        for idx, x_i in enumerate(X_train):
            linear_output = np.dot(x_i, perceptron.weights) + perceptron.bias
            y_pred = perceptron.activation(linear_output)
            error = y_train[idx] - y_pred
            update = perceptron.learning_rate * error
            perceptron.weights += update * x_i
            perceptron.bias += update
            errors += int(update != 0.0)
        perceptron.errors_history.append(errors)

        # Validação
        val_preds = perceptron.predict(X_val)
        val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Parou na época {epoch+1} (sem melhora em {patience} épocas). Melhor val_acc={best_val_acc:.2f}")
            break

    return perceptron, best_val_acc

class_sep_values = np.linspace(0.5, 3.0, 6)
flip_y_values = np.linspace(0, 0.2, 5)

results = []

for class_sep in class_sep_values:
    for flip_y in flip_y_values:
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=42
        )
        perceptron, best_val_acc = train_with_early_stopping(X, y, patience=5, n_epochs=100, learning_rate=0.1)
        preds = perceptron.predict(X)
        acc = (preds == y).mean()
        results.append((class_sep, flip_y, acc))
        print(f"class_sep={class_sep:.2f}, flip_y={flip_y:.2f}, acc treino+val={acc*100:.2f}%, melhor val={best_val_acc*100:.2f}%")

# Visualização
df = pd.DataFrame(results, columns=['class_sep', 'flip_y', 'accuracy'])
pivot = df.pivot(index='class_sep', columns='flip_y', values='accuracy')

plt.figure(figsize=(8,6))
plt.imshow(pivot, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Accuracy')
plt.xticks(range(len(flip_y_values)), [f"{v:.2f}" for v in flip_y_values])
plt.yticks(range(len(class_sep_values)), [f"{v:.2f}" for v in class_sep_values])
plt.xlabel('flip_y')
plt.ylabel('class_sep')
plt.title('Perceptron Accuracy vs. class_sep & flip_y (com early stopping)')
plt.show()