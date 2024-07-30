import os

# Instalando biblioteca do MLflow
os.system('pip install mlflow --quiet')

# Exibir a URL do MLflow UI
print("\n\nMLflow UI está rodando em http://localhost:5000\n")
print("Caso queria cancelar a execução do MLflow, pressione Ctrl + C no terminal.\n\n\n\n")
print("Estabelecendo conexão do servidor local com o MLflow UI...")
os.system('mlflow ui')
