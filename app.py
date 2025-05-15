# app.py

'''
📁 xai_app/
│
├── app.py                      # [x] Principal app Streamlit
├── modelo.pkl                  # Modelo treinado salvo (via pickle)
├── explicadores.py             # SHAP, LIME e outros modularizados
├── utils.py                    # Funções auxiliares (preprocessamento, visualizações, etc.)
├── requirements.txt            # Dependências
└── Procfile                    # Necessário para deploy no Heroku
'''

import streamlit as st
import pandas as pd
from utils import (
    carregar_dados,
    listar_modelos,
    inferir_scaler_nome,
    carregar_modelo_e_scaler,
    preprocessar_dados,
    avaliar_modelo
)
from explicadores import explicar_modelo

st.set_page_config(page_title="XAI Genérico", layout="wide")
st.title("🧠 Predição com XAI — Plataforma Flexível")

# 📦 Seleção de modelo
st.sidebar.header("⚙️ Selecione o Modelo")
modelos_disponiveis = listar_modelos()
modelo_escolhido = st.sidebar.selectbox("Modelos disponíveis:", modelos_disponiveis)

# 📂 Upload de dados
st.sidebar.header("📂 Upload dos Dados")
arquivo = st.sidebar.file_uploader("Selecione um arquivo CSV", type=["csv"])

# 🔄 Interação principal
if arquivo and modelo_escolhido:
    df = carregar_dados(arquivo)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    st.subheader("📋 Pré-visualização dos Dados")
    st.dataframe(df.head())

    nome_scaler = inferir_scaler_nome(modelo_escolhido)
    modelo, scaler = carregar_modelo_e_scaler(modelo_escolhido, nome_scaler)

    if st.button("🔍 Realizar Predição"):
        X_proc = preprocessar_dados(df, scaler)
        pred = modelo.predict(X_proc)
        df_resultado = df.copy()
        df_resultado["Predição"] = pred

        st.subheader("📊 Resultados")
        st.dataframe(df_resultado)

        if "target" in df.columns:
            st.subheader("📈 Avaliação do Modelo")
            relatorio, matriz = avaliar_modelo(df["target"], pred)
            st.code(relatorio)
            st.pyplot(matriz)

        metodo = st.radio("Método de explicação", ["lime"])
        explicar_modelo(modelo, X_proc, metodo=metodo, nomes_features=df.columns.tolist())

        st.download_button("📥 Baixar Resultados", df_resultado.to_csv(index=False), file_name="resultados.csv")
else:
    st.info("Carregue um arquivo e selecione um modelo para iniciar.")
