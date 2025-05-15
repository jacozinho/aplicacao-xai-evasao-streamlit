# Procfile

'''
📁 xai_app/
│
├── app.py                      # Principal app Streamlit
├── modelo.pkl                  # Modelo treinado salvo (via pickle)
├── explicadores.py             # SHAP, LIME e outros modularizados
├── utils.py                    # Funções auxiliares (preprocessamento, visualizações, etc.)
├── requirements.txt            # Dependências
└── Procfile                    # [x] Necessário para deploy no Heroku
'''

web: streamlit run app.py --server.port=$PORT --server.enableCORS=false