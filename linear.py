import numpy as np
import matplotlib.pyplot as plt

# Crie dois grupos de pontos bem separados
np.random.seed(42)
# Classe 0: centro em (-2, -2)
class_0 = np.random.randn(50, 2) + [-2, -2]
# Classe 1: centro em (2, 2)
class_1 = np.random.randn(50, 2) + [2, 2]
X = np.vstack([class_0, class_1])
y = np.hstack([np.zeros(50), np.ones(50)])

# Treinamento do perceptron
w = np.zeros(2)
b = 0
lr = 0.1
for _ in range(100):
  for xi, yi in zip(X, y):
    y_pred = 1 if np.dot(w, xi) + b > 0 else 0
    w += lr * (yi - y_pred) * xi
    b += lr * (yi - y_pred)

# Equação da reta de decisão: w[0]*x + w[1]*y + b = 0
x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
y_vals = -(w[0]*x_vals + b)/w[1]

# Plot dos pontos e da reta de decisão
plt.scatter(class_0[:,0], class_0[:,1], color='blue', label='Classe 0')
plt.scatter(class_1[:,0], class_1[:,1], color='red', label='Classe 1')
plt.plot(x_vals, y_vals, 'k--', label='Reta de decisão')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron: Reta de decisão')
plt.show()

# Verificação dos lados corretos
preds = np.array([1 if np.dot(w, xi) + b > 0 else 0 for xi in X])
print("Todos os pontos classificados corretamente?", np.all(preds == y))

# Experimente mover os centros mais próximos
class_0_close = np.random.randn(50, 2) + [-1, -1]
class_1_close = np.random.randn(50, 2) + [1, 1]
X_close = np.vstack([class_0_close, class_1_close])
y_close = np.hstack([np.zeros(50), np.ones(50)])

w = np.zeros(2)
b = 0
for _ in range(100):
  for xi, yi in zip(X_close, y_close):
    y_pred = 1 if np.dot(w, xi) + b > 0 else 0
    w += lr * (yi - y_pred) * xi
    b += lr * (yi - y_pred)

preds_close = np.array([1 if np.dot(w, xi) + b > 0 else 0 for xi in X_close])
print("Com centros próximos, todos classificados corretamente?", np.all(preds_close == y_close))