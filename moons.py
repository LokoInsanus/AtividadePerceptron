from sklearn.datasets import make_moons
from perceptron import Perceptron

X, y = make_moons(
  n_samples=200,
  noise=0.15, # Adiciona ruído realista
  random_state=42
)

# Cria e treina o perceptron
perceptron = Perceptron(learning_rate=0.1, n_epochs=10)
perceptron.fit(X, y)

# Faz previsões
predictions = perceptron.predict(X)
print("Predictions:", predictions)

# Avalia a acurácia
accuracy = (predictions == y).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualização
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moons Dataset - Perceptron Classification')
plt.show()