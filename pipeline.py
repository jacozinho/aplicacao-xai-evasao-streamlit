# pipeline.py

import subprocess
import streamlit.web.bootstrap
import os

# Caminho dos scripts
CAMINHO_BATCH = "executar_batch_modelos.py"
CAMINHO_APP = "app.py"

# Etapa 1: Treinar os modelos
print("🔄 Iniciando o treinamento de múltiplos modelos...")
subprocess.run(["python", CAMINHO_BATCH], check=False)
print("✅ Modelos treinados com sucesso!")

# Etapa 2: Iniciar o app Streamlit
print("🚀 Iniciando a aplicação Streamlit...")

# Executa o Streamlit diretamente dentro do Python
# Alternativamente, poderia usar subprocess com: 
subprocess.run(["streamlit", "run", CAMINHO_APP])
#streamlit.web.bootstrap.run(CAMINHO_APP, [], {})

'''
from streamlit import config as _config
from streamlit.web.bootstrap import run
_config.set_option("server.headless", True)
run('your_app.py', args=[], flag_options=[], is_hello=False)
'''
