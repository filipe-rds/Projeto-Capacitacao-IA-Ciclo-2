# -*- coding: utf-8 -*-
"""
Projeto_Final_IA_(Ciclo_2).ipynb


# 📜 Projeto Final - Capacitação IA (Ciclo 2)
# 🎓 Aluno: Filipe da Silva Rodrigues

## 💻 Bibliotecas Necessárias
"""

# Instalação de bibliotecas necessárias para execução do código
import os
os.system('pip install numpy pandas scikit-learn mlflow xgboost lightgbm --quiet')

# Tratamento de Dataset e Métricas
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# Modelos de Treinamento
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
# Multi-layer Perceptron (MLP)
from sklearn.neural_network import MLPRegressor
# Support Vector Machine
from sklearn.svm import SVR
# Random Forest, Bagging e Gradient Boosting
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
# XGBoost
from xgboost import  XGBRegressor
# LightGBM
from lightgbm import LGBMRegressor

# Armazenamento e Análise de Modelos
import mlflow
import mlflow.sklearn

# Terminal
import warnings
from IPython.display import clear_output
warnings.filterwarnings("ignore")

"""
---

👾 **Dataset de Regressão - Hugging Face: Einstellung/demo-salaries**

Esse dataframe é um conjunto de dados que contém informações sobre salários e características de diferentes cargos na área de ciência de dados. As variáveis são:

- `work_year`: o ano em que o salário foi reportado (ex: 2023).
- `experience_level`: o nível de experiência do funcionário (EN = Júnior, MI = Pleno, SE = Sênior, EX = Executivo).
- `employment_type`: o tipo de emprego (PT = Meio período, FT = Tempo integral, CT = Contrato, FL = Freelance).
- `job_title`: o título do cargo do funcionário (ex: Data Scientist, Data Engineer).
- `salary`: o salário anual bruto reportado.
- `salary_currency`: a moeda na qual o salário foi pago (ex: USD, EUR).
- `salary_in_usd`: o salário anual bruto convertido para USD.
- `employee_residence`: o país de residência do funcionário (ex: US, CA, GB).
- `remote_ratio`: a proporção de trabalho remoto (0 = Presencial, 50 = Híbrido, 100 = Totalmente remoto).
- `company_location`: o país onde a empresa está localizada.
- `company_size`: o tamanho da empresa (S = Pequena, M = Média, L = Grande).

✅ **Objetivo:** Prever qual salário anual em USD de um funcionário de acordo com as características coletadas.

---
"""

# Carregar o dataset
url = 'https://huggingface.co/datasets/Einstellung/demo-salaries/resolve/main/ds_salaries.csv'
dataset = pd.read_csv(url)

# Analisar o dataset
print('\nInformações do Dataset:\n')
print(dataset.info())

print('\nVerificar Valores Nulos:\n')
print(dataset.isnull().sum())

print('\nVerificar Valores Únicos em Features Categóricas:\n')
for col in dataset.select_dtypes(include=['object']).columns:
    print(f'{col}: {dataset[col].nunique()} unique values')

# Exibir o dataset original
print('\nDataset Original:\n')
print(dataset)

# Criar uma cópia do dataset para efetuar os devidos tratamentos
df = dataset.copy()

# Normalizando os dados das features na escala (0..1)
columns_to_normalize = ['salary', 'remote_ratio', 'work_year']
df[columns_to_normalize] = MinMaxScaler().fit_transform(df[columns_to_normalize])

# Separar os dados para o tratamento de features categóricas
target = df['salary_in_usd'].copy()
features = df.drop('salary_in_usd', axis=1).copy()

# Convertendo features categóricas para números com OneHotEncoder
categorical_columns = ['experience_level', 'employment_type', 'job_title', 'salary_currency',
                       'employee_residence', 'company_location', 'company_size']

column_transform = make_column_transformer(
    (OneHotEncoder(drop='first'), categorical_columns), remainder='passthrough')

# Transformando os dados
features_transformed = column_transform.fit_transform(features)
columns_names = column_transform.get_feature_names_out()

# Transformando o resultado em um DataFrame
features_transformed_df = pd.DataFrame(
    data=features_transformed.toarray(), columns=columns_names)

# Dicionário para mapear as colunas a serem renomeadas
rename_mapping = {col: col.replace('onehotencoder__', '').replace('remainder__', '')
                  for col in features_transformed_df.columns}

# Renomeando as colunas
features_transformed_df.rename(columns=rename_mapping, inplace=True)

# Combinando as features transformadas com o target
df = pd.concat(
    [features_transformed_df, target.reset_index(drop=True)], axis=1)

# Exibindo o DataFrame tratado com as colunas renomeadas
print('\nDataset Tratado para Treinamento:\n')
print(df)

# Análise de correlação entre as features
print('\nMatriz de Correlação:\n')
correlation_matrix = df.corr()
print(correlation_matrix)

# Separando os dados
y = df['salary_in_usd']  # Coluna 'salary_in_usd'
x = df.drop('salary_in_usd', axis=1)  # Todas as outras colunas

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)

# Aplicar Recursive Filter Elimination (RFE) para Seleção de Features
num_features = 50  # Número de features a serem selecionadas
estimator = DecisionTreeRegressor()
# estimator = LogisticRegression(max_iter=1000)
# estimator = Lasso(alpha=0.01, max_iter=1000)
selector = RFE(estimator, n_features_to_select=num_features)

# Fit e Transform no Conjunto de Treinamento
x_train_selected = selector.fit_transform(x_train, y_train)

# Transform no Conjunto de Teste
x_test_selected = selector.transform(x_test)

"""## 🧪 Experimentos no MLFLOW"""

# Definir os modelos e suas variações de parâmetros
models = {
    "DecisionTree": [
        {"criterion": "squared_error", "max_depth": 10, "min_samples_split": 4},
        {"criterion": "squared_error", "max_depth": 20, "min_samples_split": 10},
        {"criterion": "friedman_mse", "max_depth": 15, "min_samples_split": 5},
    ],
    "SVR": [
        {"C": 1.0, "kernel": "linear", "epsilon": 0.1},
        {"C": 10.0, "kernel": "rbf", "epsilon": 0.01},
        {"C": 100.0, "kernel": "poly", "degree": 3, "epsilon": 0.001},
    ],
    "MLPRegressor": [
        {"hidden_layer_sizes": (100, 50), "activation": "relu",
         "solver": "adam", "max_iter": 1000},
        {"hidden_layer_sizes": (50, 50, 50), "activation": "tanh",
         "solver": "adam", "max_iter": 1000},
        {"hidden_layer_sizes": (100, 50), "activation": "relu",
         "solver": "lbfgs", "max_iter": 1000},
    ],
    "Bagging": [
        {},
    ],
    "RandomForest": [
        {},
    ],
    "GradientBoosting": [
        {},
    ],
    "XGBoost": [
        {},
    ],
    "LightGBM": [
        {},
    ]
}


# Mapeamento de nomes de modelos para classes 
model_classes = {
    "DecisionTree": DecisionTreeRegressor,
    "SVR": SVR,
    "MLPRegressor": MLPRegressor,
    "Bagging": BaggingRegressor,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
}

# Preparar o ambiente do MLFlow e início do experimento

# lista para armazenar os resultados
results = []

# Iniciar o experimento
mlflow.set_experiment("exp_projeto_ciclo_2")

# Contador para evitar conflitos de nomes
counter = 0

print("\n\nIniciando experimento no MLflow\n")
print("Modelos de regressão que serão treinados:\n")
for model_name, param_variations in models.items():
    for params in param_variations:
        print(f"{model_name} - Parâmetros: {params}")
        
print("\nSerão utilizados 10 folds para validação cruzada.")
print("Serão selecionados os 3 melhores modelos com base na métrica MAPE.")
print("Aguarde, isso pode levar alguns minutos...")

# Run principal
with mlflow.start_run(run_name="Projeto Final Ciclo 2") as main_run:
    for model_name, param_variations in models.items():
        for params in param_variations:
            counter += 1
            # Run aninhada
            with mlflow.start_run(run_name=f"{counter}. {model_name}", nested=True):
                # Instanciar o modelo usando o dicionário de classes
                model = model_classes[model_name](**params)

                # Realizar validação cruzada para 10 folds e calcular as previsões
                predictions = cross_val_predict(
                    model, x_train_selected, y_train, cv=10)

                # Calcular as métricas
                rmse = np.sqrt(mean_squared_error(y_train, predictions))
                mae = mean_absolute_error(y_train, predictions)
                mape = np.mean(np.abs((y_train - predictions) / y_train)) * 100

                # Registrar os parâmetros
                mlflow.log_param("model_name", model_name)

                # Registrar parâmetros individualmente
                for key, value in params.items():
                    mlflow.log_param(key, str(value))

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("MAPE", mape)
                mlflow.sklearn.log_model(model, f"{model_name}")

                # Armazenar resultados
                results.append({"model": model_name, "params": params,
                               "RMSE": rmse, "MAE": mae, "MAPE": mape})

    # Selecionar os 3 melhores modelos com base na métrica MAPE
    best_models = sorted(results, key=lambda x: x["MAPE"])[:3]

    # Limpar a saída do terminal
    clear_output(wait=True)

    # Exibir os melhores modelos e suas métricas
    print("\nMelhores Modelos:\n")
    for model_info in best_models:
        print(model_info)

"""## 💾 Modelos Registrados no MLFLOW"""

# Definir o tracking URI do MLfloww
mlflow_tracking_uri = 'http://localhost:5000'
mlflow.set_tracking_uri(mlflow_tracking_uri)


# Exibir a URL do MLflow UI
print(f"\n\nMLflow UI está rodando em {mlflow_tracking_uri}\n")
print("Caso queria cancelar a execução do MLflow, pressione Ctrl + C no terminal.\n\n\n\n")
print("Estabelecendo conexão do servidor local com o MLflow UI...")
os.system('mlflow ui')
