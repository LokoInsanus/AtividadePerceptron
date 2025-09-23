# Atividade Prática de Inteligência Artifical

## Dupla

- Marcelo de Oliveira Costa Pereira
-

---

## Relatórios dos Experimentos

### iris.py

1. **Descrição do Dataset**
   - Número de amostras e features: 100 amostras, 2 features (comprimento da sépala e comprimento da pétala)
   - Distribuição das classes: 50 Setosa, 50 Versicolor
   - É linearmente separável? Sim (para Setosa vs Versicolor)

2. **Resultados**
   - Acurácia no treino e teste: ~100% (com as features escolhidas)
   - Número de épocas até convergência: 10
   - Tempo de treinamento: <1s

3. **Visualizações**
   - Gráfico de convergência: Não implementado
   - Regiões de decisão: Visualização dos pontos do teste coloridos por classe
   - Matriz de confusão: Não implementado

4. **Análise**
   - O perceptron foi adequado para este problema? Sim, pois as classes são linearmente separáveis
   - Que melhorias você sugeriria? Adicionar matriz de confusão e gráfico de convergência
   - Comparação com suas expectativas: Resultado esperado, alta acurácia

---

### moons.py

1. **Descrição do Dataset**
   - Número de amostras e features: 200 amostras, 2 features
   - Distribuição das classes: Balanceada
   - É linearmente separável? Não

2. **Resultados**
   - Acurácia no treino e teste: ~75% (varia conforme ruído)
   - Número de épocas até convergência: 10
   - Tempo de treinamento: <1s

3. **Visualizações**
   - Gráfico de convergência: Não implementado
   - Regiões de decisão: Visualização dos pontos coloridos por classe
   - Matriz de confusão: Não implementado

4. **Análise**
   - O perceptron foi adequado para este problema? Não, pois o dataset não é linearmente separável
   - Que melhorias você sugeriria? Usar modelos não lineares (MLP, SVM com kernel)
   - Comparação com suas expectativas: Acurácia baixa, como esperado

---

### cancer.py

1. **Descrição do Dataset**
   - Número de amostras e features: 569 amostras, 2 ou 30 features (dependendo da versão)
   - Distribuição das classes: 357 benignos, 212 malignos
   - É linearmente separável? Parcialmente

2. **Resultados**
   - Acurácia no treino e teste: ~63% (2 features), ~96% (com PCA/30 features)
   - Número de épocas até convergência: 100
   - Tempo de treinamento: <2s

3. **Visualizações**
   - Gráfico de convergência: Não implementado
   - Regiões de decisão: Visualização dos pontos do teste (2D ou PCA)
   - Matriz de confusão: Não implementado

4. **Análise**
   - O perceptron foi adequado para este problema? Parcialmente, melhor com mais features
   - Que melhorias você sugeriria? Usar todas as features ou modelos mais complexos
   - Comparação com suas expectativas: Melhor resultado com PCA, como esperado

---

### ruido.py

1. **Descrição do Dataset**
   - Número de amostras e features: 200 amostras, 2 features
   - Distribuição das classes: Balanceada, com ruído controlado
   - É linearmente separável? Depende dos parâmetros (class_sep, flip_y)

2. **Resultados**
   - Acurácia no treino e teste: Varia de ~50% (muito ruído) até ~100% (alta separação)
   - Número de épocas até convergência: Até 100 (com early stopping)
   - Tempo de treinamento: <2s por experimento

3. **Visualizações**
   - Gráfico de convergência: Não implementado
   - Regiões de decisão: Não implementado
   - Matriz de confusão: Não implementado
   - Heatmap de acurácia vs. separação/ruído

4. **Análise**
   - O perceptron foi adequado para este problema? Só quando há separação clara e pouco ruído
   - Que melhorias você sugeriria? Usar regularização ou modelos não lineares
   - Comparação com suas expectativas: Resultado esperado, acurácia cai com ruído

---

### linear.py

1. **Descrição do Dataset**
   - Número de amostras e features: 100 amostras, 2 features
   - Distribuição das classes: 50 classe 0, 50 classe 1
   - É linearmente separável? Sim (primeiro experimento), não (quando centros próximos)

2. **Resultados**
   - Acurácia no treino e teste: 100% (centros separados), <100% (centros próximos)
   - Número de épocas até convergência: 100
   - Tempo de treinamento: <1s

3. **Visualizações**
   - Gráfico de convergência: Não implementado
   - Regiões de decisão: Reta de decisão plotada
   - Matriz de confusão: Não implementado

4. **Análise**
   - O perceptron foi adequado para este problema? Sim, quando linearmente separável
   - Que melhorias você sugeriria? Testar com mais ruído ou menos separação
   - Comparação com suas expectativas: Resultado esperado, perceptron falha com dados não separáveis