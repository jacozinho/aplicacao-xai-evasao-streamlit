# executar_batch_modelos.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from treinar_modelo import treinar_e_salvar


def carregar_dados_para_treinamento(df_X_path, df_y_path):
    """Carrega os dados de treinamento a partir dos arquivos CSV."""
    df_X = pd.read_csv(df_X_path)
    df_y = pd.read_csv(df_y_path)

    # Remove coluna desnecessária, se existir
    for df in [df_X, df_y]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

    return df_X, df_y.squeeze()  # Garante que y seja uma Series


def main():
    print("🚀 Iniciando treinamento de modelos...")

    # Caminhos fixos dos datasets (pode ser parametrizado futuramente)
    diretorio = "datasets/"
    df_X_train_path = os.path.join(diretorio, "2024-01-04_X_train_up.csv")
    df_y_train_path = os.path.join(diretorio, "2024-01-04_y_train_up.csv")

    # Validação de existência dos arquivos
    if not os.path.exists(df_X_train_path) or not os.path.exists(df_y_train_path):
        print("❌ Arquivos de treino não encontrados.")
        return

    # Carrega os dados
    X, y = carregar_dados_para_treinamento(df_X_train_path, df_y_train_path)

    # Define os modelos a treinar
    modelos = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "svc": SVC(probability=True),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    # Treina e salva cada modelo
    for nome, modelo in modelos.items():
        print(f"🔧 Treinando modelo: {nome}")
        treinar_e_salvar(X, y, nome, modelo)
        print(f"✅ Modelo '{nome}' salvo com sucesso!")

    print("🏁 Treinamento concluído.")


if __name__ == "__main__":
    main()
