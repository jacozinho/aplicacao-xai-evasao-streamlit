# executar_batch_modelos.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from treinar_modelo import treinar_e_salvar

def carregar_dados_para_treinamento(df_X_train, df_y_train):
    df_X = pd.read_csv(df_X_train)
    df_y = pd.read_csv(df_y_train)
    # X = df.drop(columns=[coluna_alvo])

    df_X.drop('Unnamed: 0', axis=1, inplace=True)
    df_y.drop('Unnamed: 0', axis=1, inplace=True)

    X = df_X
    y = df_y

    return X, y


if __name__ == "__main__":
    print('executa modelos 1')
    # Caminho e coluna alvo definidos fixamente (pode ser parametrizado com argparse futuramente)
    diretorio = "datasets/"
    df_X_train = diretorio+"2024-01-04_X_train_up.csv"
    df_y_train = diretorio+"2024-01-04_y_train_up.csv" #"situacao"

    X, y = carregar_dados_para_treinamento(df_X_train, df_y_train)

    modelos = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "svc": SVC(probability=True),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    for nome, modelo in modelos.items():
        treinar_e_salvar(X, y, nome, modelo)
