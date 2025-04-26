import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Diagnóstico de Doença Cardíaca", layout="centered")

# Sidebar - Upload de dados
st.sidebar.title("📁 Carregamento de Dados")
uploaded_file = st.sidebar.file_uploader("Faça upload do seu arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).drop_duplicates()

    st.sidebar.success("✅ Dados carregados com sucesso!")

    # Sidebar - Seleção de variáveis
    st.sidebar.title("⚙️ Configurações")
    all_columns = df.columns.tolist()
    default_target = "target" if "target" in all_columns else all_columns[0]
    target_var = st.sidebar.selectbox("Selecione a variável alvo (target):", all_columns, index=all_columns.index("target"))

    input_features = st.sidebar.multiselect(
        "Selecione as variáveis preditoras (features):",
        [col for col in all_columns if col != target_var],
        default=[col for col in all_columns if col != target_var]
    )

    st.title("Diagnóstico de Doença Cardíaca com Machine Learning")

    if st.checkbox("Mostrar primeiros dados"):
        st.dataframe(df.head())

    if input_features and target_var:
        # Separar dados
        X = df[input_features]
        y = df[target_var]

        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=415)

        # Modelos
        models = {
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True)
        }

        model_choice = st.selectbox("Escolha o modelo de classificação:", list(models.keys()))
        model = models[model_choice]

        # Treinamento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Resultados
        st.subheader("Relatório de Classificação")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Matriz de Confusão")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"{model_choice} (AUC = {roc_auc:.2f})")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel("Taxa de Falsos Positivos")
        ax2.set_ylabel("Taxa de Verdadeiros Positivos")
        ax2.set_title("Curva ROC")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    else:
        st.warning("Por favor, selecione a variável alvo e ao menos uma variável preditora.")
else:
    st.title(" Diagnóstico de Doença Cardíaca com Machine Learning")

    st.markdown("Este aplicativo interativo utiliza algoritmos de aprendizado de máquina para auxiliar na **predição de doenças cardíacas** com base em dados clínicos.")
    st.info("Faça upload de um arquivo CSV com os dados na barra lateral para começar.")
