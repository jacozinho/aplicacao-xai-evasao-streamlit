# explicadores.py

'''
📁 xai_app/
│
├── app.py                      # Principal app Streamlit
├── modelo.pkl                  # Modelo treinado salvo (via pickle)
├── explicadores.py             # [x] SHAP, LIME e outros modularizados
├── utils.py                    # Funções auxiliares (preprocessamento, visualizações, etc.)
├── requirements.txt            # Dependências
└── Procfile                    # Necessário para deploy no Heroku
'''

import streamlit as st
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# st.set_option("deprecation.showPyplotGlobalUse", False)

def explicar_modelo(modelo, X, metodo="shap", nomes_features=None):
    if metodo == "shap":
        st.markdown("### 🔍 Explicação com SHAP")
        
        try:
            explainer = shap.Explainer(modelo, X)
            shap_values = explainer(X)
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot()
        except Exception as e:
            st.error(f"Erro com SHAP: {e}")        
    elif metodo == "lime":
        st.markdown("### 🔍 Explicação com LIME")
        if nomes_features is None:
            st.warning("Nomes de features são obrigatórios para LIME.")
            return
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X,
                feature_names=nomes_features,
                mode="classification"
            )
            i = st.number_input("Índice da instância para explicar", min_value=0, max_value=len(X)-1, step=1)
            exp = explainer.explain_instance(X[i], modelo.predict_proba)
            st.components.v1.html(exp.as_html(), height=500, scrolling=True)
        except Exception as e:
            st.error(f"Erro com LIME: {e}")
    else:
        st.warning("Método não reconhecido.")
