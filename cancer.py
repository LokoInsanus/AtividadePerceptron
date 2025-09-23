from sklearn.datasets import load_breast_cancer
from perceptron import Perceptron
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def versao_a():
  cancer = load_breast_cancer()
  X = cancer.data[:, [0, 1]]
  y = cancer.target
  print(f"Features: {cancer.feature_names[[0, 1]]}")
  print(f"Classes: {cancer.target_names}")

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  perceptron = Perceptron(learning_rate=0.01, n_epochs=100)
  perceptron.fit(X_train, y_train)
  predictions = perceptron.predict(X_test)
  accuracy = (predictions == y_test).mean()
  print("Predictions:", predictions)
  print("True labels:", y_test)
  print(f"Accuracy: {accuracy * 100:.2f}%")

  plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
  plt.xlabel(cancer.feature_names[0])
  plt.ylabel(cancer.feature_names[1])
  plt.title('Breast Cancer Dataset - Perceptron Classification (Test Set)')
  plt.show()

def versao_b():
  from sklearn.decomposition import PCA
  cancer = load_breast_cancer()
  X = cancer.data
  y = cancer.target
  print(f"Features: {cancer.feature_names}")
  print(f"Classes: {cancer.target_names}")

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  perceptron = Perceptron(learning_rate=0.01, n_epochs=100)
  perceptron.fit(X_train, y_train)
  predictions = perceptron.predict(X_test)
  accuracy = (predictions == y_test).mean()
  print("Predictions:", predictions)
  print("True labels:", y_test)
  print(f"Accuracy: {accuracy * 100:.2f}%")

  pca = PCA(n_components=2)
  X_test_2d = pca.fit_transform(X_test)
  plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='viridis', edgecolors='k')
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.title('Breast Cancer Dataset - Perceptron Classification (Test Set, PCA Reduced)')
  plt.show()

if __name__ == "__main__":
  escolha = input("Versão A ou B? ").strip().upper()
  if escolha == "A":
    versao_a()
  elif escolha == "B":
    versao_b()
  else:
    print("Escolha inválida. Digite 'A' ou 'B'.")