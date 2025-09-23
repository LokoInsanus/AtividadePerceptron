import matplotlib.pyplot as plt
from sklearn import datasets
from perceptron import Perceptron
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# IMPORTANTE: Use apenas as classes 0 e 1 (Setosa e Versicolor)
# Classe 2 (Virginica) não é linearmente separável das outras
mask = iris.target != 2
X = iris.data[mask]
y = iris.target[mask]

# Sugestão: Use apenas 2 features para visualização
# Por exemplo: índices [0, 2] = comprimento da sépala e comprimento da pétala
X = X[:, [0, 2]]

# Divide em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cria e treina o perceptron
perceptron = Perceptron(learning_rate=0.1, n_epochs=10)
perceptron.fit(X_train, y_train)

# Faz previsões
predictions = perceptron.predict(X_test)
print("Predictions:", predictions)
print("True labels:", y_test)

# Avalia a acurácia
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualização
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Iris Dataset - Perceptron Classification (Test Set)')
plt.show()