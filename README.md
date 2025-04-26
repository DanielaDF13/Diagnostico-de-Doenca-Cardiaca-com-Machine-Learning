# Diagnóstico de Doença Cardíaca com Machine Learning

Este projeto oferece uma aplicação interativa utilizando **Streamlit** e **modelos de aprendizado de máquina** para prever a presença de doenças cardíacas com base em dados clínicos. 

## Objetivo

A doença cardíaca é uma das principais causas de morte no mundo. Diagnósticos precoces e precisos podem salvar vidas e melhorar a qualidade de vida dos pacientes. O objetivo deste projeto é auxiliar profissionais de saúde no **diagnóstico precoce de doenças cardíacas** por meio de técnicas de aprendizado de máquina.

A base de dados utilizada neste projeto é proveniente do repositório UCI Machine Learning Repository, contendo informações clínicas e demográficas de pacientes. Com base nessas informações, pretende-se desenvolver um modelo capaz de prever a **presença ou ausência de doença cardíaca**.

Essa previsão pode ser uma ferramenta valiosa para:
- Apoiar decisões médicas;
- Identificar pacientes de alto risco;
- Direcionar exames e recursos de forma mais eficaz.

---

### Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Streamlit

---

### Funcionalidades do Streamlit

- **Upload de datasets customizados (.csv)**: Permite que o usuário faça o upload de arquivos CSV para análise, com a possibilidade de carregar dados de diferentes fontes.

- **Seleção de variável alvo e features**: O usuário pode escolher qual coluna do dataset será a variável alvo e quais colunas serão usadas como features para o modelo, diretamente pela interface da barra lateral.

- **Treinamento com 4 modelos**:
  - **Logistic Regression**: Um modelo linear simples para classificação binária.
  - **KNN (K-Nearest Neighbors)**: Um modelo baseado em vizinhos mais próximos para prever a classe.
  - **Random Forest**: Um modelo de ensemble baseado em múltiplas árvores de decisão, adequado para problemas de classificação.
  - **SVM (Support Vector Machine)**: Um modelo que busca encontrar a melhor linha ou hiperplano para separar as classes.
    
- **Exibição do relatório de classificação, matriz de confusão e curva ROC**:
  - **Relatório de Classificação**: Apresenta métricas como precisão, recall e f1-score para avaliar a performance de cada modelo.
  - **Matriz de Confusão**: Exibe o número de falsos positivos, falsos negativos, verdadeiros positivos e verdadeiros negativos.
  - **Curva ROC**: Mostra o desempenho de cada modelo ao longo de diferentes limiares de decisão, destacando o AUC (Área sob a Curva).

