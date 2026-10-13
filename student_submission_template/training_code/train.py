import pandas as pd
import numpy as np

## Parte 1: Colocar os dados do ficheiro de excel numa database
df = pd.read_excel(
"data/chlorophyll_student_2015_2023.xlsx",
sheet_name="daily_data"
)

df["date"] = pd.to_datetime(df["date"]) # Transformar as strings da date em variaveis de datas especificas do pandas 
df = df.sort_values("date").reset_index(drop=True) # Ordena a database de forma a que os dias e respetivas variáveis estejam por ordem cronologica

exog_columns = list(df.columns[2:])   # só as 12 exógenas

def build_features(df, P, Q):
    copy_df = df.copy()
    feats = {}

    # lags da clorofila: 1..P
    for i in range(1, P + 1):
        feats[f"chl_lag_{i}"] = copy_df["chlorophyll_a_mg_m3"].shift(i)

    # lags das exógenas: 0..Q
    for col in exog_columns:
        for j in range(0, Q + 1):
            feats[f"{col}_lag_{j}"] = copy_df[col].shift(j)

    # sazonalidade
    doy = copy_df["date"].dt.dayofyear
    feats["s_year"] = np.sin(2 * np.pi * doy / 365.25)
    feats["c_year"] = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame(feats, index=copy_df.index)


X = build_features(df, 3, 7) 
valid = X.notna().all(axis=1) # Dá um vetor coluna com true se todas as features forem válidas e false se houver algum NaN 
X = X[valid] # Filtra a database para só ficar com as linhas válidas
y = df.loc[valid, "chlorophyll_a_mg_m3"] # linhas de clorofila válidas segundo X
dates = df.loc[valid, "date"] # mesma coisa para as datas

## Separar a database em treino e teste (Data limite de teste: 2022-01-01)
train_mask = dates < "2022-01-01"
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_val = X.loc[~train_mask]
y_val = y.loc[~train_mask]

# Calcular o erro do pior caso (baseline) para depois comparar com o modelo
import sklearn.metrics as metrics
def metrics_calc(y_true, y_pred):
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    print("Metrics: R2 - " + str(r2) + ", MSE - " + str(mse) + ", RMSE - " + str(rmse) + ", MAE - " + str(mae))


# Calcular o erro do pior caso (baseline) para depois comparar com o modelo
metrics_calc(y_val, X_val["chl_lag_1"])

# Modelo de regressão linear
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr_fit = model_lr.fit(X_train, y_train)
model_lr_pred = model_lr.predict(X_val)
metrics_calc(y_val, model_lr_pred)