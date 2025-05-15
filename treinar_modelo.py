# modelo.py

'''
📁 xai_app/
│
├── app.py                     # Principal app Streamlit
├── modelo.py                  # [x] Modelo treinado salvo (via pickle)
├── explicadores.py            # SHAP, LIME e outros modularizados
├── utils.py                   # Funções auxiliares (preprocessamento, visualizações, etc.)
├── requirements.txt           # Dependências
└── Procfile                   # Necessário para deploy no Heroku
└── /datasets                  # Diretório com os conjuntos de dados
└── /pkls                      # Diretório com os modelos treinados
'''

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def treinar_e_salvar(X, y, nome, modelo):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo.fit(X_scaled, y.values.ravel())
    # modelo.fit(X_scaled, y)

    diretorio = "pkls/"
    with open(f"{diretorio}modelo_{nome}.pkl", "wb") as f: pickle.dump(modelo, f)
    with open(f"{diretorio}scaler_{nome}.pkl", "wb") as f: pickle.dump(scaler, f)
    print(f"✅ modelo_{nome}.pkl e scaler_{nome}.pkl salvos.")

'''
if __name__ == "__main__":
    df = pd.read_csv("seus_dados.csv")
    target = "target"
    X, y = df.drop(columns=[target]), df[target]

    modelos = {
        "rf": RandomForestClassifier(n_estimators=100),
        "logreg": LogisticRegression(max_iter=1000),
        "svc": SVC(probability=True),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    for nome, modelo in modelos.items():
        treinar_e_salvar(X, y, nome, modelo)
'''