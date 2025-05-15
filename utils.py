# utils.py

'''
📁 xai_app/
│
├── app.py                      # Principal app Streamlit
├── modelo.pkl                  # Modelo treinado salvo (via pickle)
├── explicadores.py             # SHAP, LIME e outros modularizados
├── utils.py                    # [x] Funções auxiliares (preprocessamento, visualizações, etc.)
├── requirements.txt            # Dependências
└── Procfile                    # Necessário para deploy no Heroku
'''

import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(arquivo):
    return pd.read_csv(arquivo)

def listar_modelos(diretorio="pkls/"):
    return sorted([f for f in os.listdir(diretorio)
                   if f.startswith("modelo_") and f.endswith(".pkl")])

def inferir_scaler_nome(modelo_nome):
    return modelo_nome.replace("modelo_", "scaler_")

def carregar_modelo_e_scaler(nome_modelo, nome_scaler):
    diretorio="pkls/"
    with open(diretorio+nome_modelo, "rb") as f_modelo:
        modelo = pickle.load(f_modelo)
    with open(diretorio+nome_scaler, "rb") as f_scaler:
        scaler = pickle.load(f_scaler)
    return modelo, scaler

def preprocessar_dados(df, scaler):
    return scaler.transform(df)

def avaliar_modelo(y_true, y_pred):
    relatorio = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    return relatorio, fig