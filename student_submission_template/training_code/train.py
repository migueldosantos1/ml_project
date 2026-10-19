import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pickle
from pathlib import Path
import argparse

TARGET = "chlorophyll_a_mg_m3"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "chlorophyll_student_2015_2023.xlsx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model" / "model.pkl"

# Corte cronologico para separação da database em train, validation e test sets
SPLIT_DATE_VAL = "2020-01-01"
SPLIT_DATE_TEST = "2022-01-01"

# Limiar do clipping escolhido foi percentil 99.9 do treino. Comparativamente, o percentil 99 ainda apanhava valores considerados
# realistas (cerca de 3,5), enquanto o P99.9 dá clip a todos os dados acima de 4.3878 mg/m3
CLIP_QUANTILE = 0.999

# Ranges para iteracao dos modelos
P_RANGE = [1, 3, 5, 7, 9]
Q_RANGE = [0, 1, 3, 5, 7, 9]
ALPHAS_RIDGE = [0.1, 1, 10, 100, 1000, 10000]
ALPHAS_LASSO = [0.0001, 0.001, 0.01, 0.1] # Pelo termo de alpha ser um modulo em vez de um quadrado, alphas realistas são também muito mais pequenos

def load_data(path):
    df = pd.read_excel(path, sheet_name="daily_data")
    df["date"] = pd.to_datetime(df["date"])  # strings -> datetime do pandas
    # Ordem cronologica obrigatoria: o shift(k) assume que cada linha e o dia a seguir a anterior
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df, P, Q, exog_cols):
    # Esta função constroi a database com os delays desejados (P, Q)
    copy_df = df.copy() # Copia para nao alterar o df original
    feats = {}

    # Lags da clorofila comecam em 1
    for i in range(1, P + 1):
        feats[f"chl_lag_{i}"] = copy_df[TARGET].shift(i)

    # Lags das exogenas comecam em 0 porque a medicao do proprio dia ja esta disponivel
    for col in exog_cols:
        for j in range(0, Q + 1):
            feats[f"{col}_lag_{j}"] = copy_df[col].shift(j)

    # Necessário para unificar 31 dez e 01 jan
    doy = copy_df["date"].dt.dayofyear
    feats["s_year"] = np.sin(2 * np.pi * doy / 365.25)
    feats["c_year"] = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame(feats, index=copy_df.index)


def clip_target(df, upper):
    # Clip de acordo com o limiar definido
    df_clip = df.copy()
    df_clip[TARGET] = df_clip[TARGET].clip(upper=upper)
    return df_clip


def metrics_calc(y_true, y_pred, label="", Print=True):
    # Calcula as metricas para analise do modelo e das suas previsões
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  
    mae = metrics.mean_absolute_error(y_true, y_pred)
    if Print:
        print(f">>> {label} R2 - {r2:.4f}, MSE - {mse:.4f}, RMSE - {rmse:.4f}, MAE - {mae:.4f}")
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

def metricas_normalizar(X_train):
    # Retorna a média e standart dev de todas as colunas do dataset
    mu = X_train.mean()
    sd = X_train.std()
    sd = sd.replace(0, 1) #Se sd = 0 divisão daria NaN
    return mu, sd


def splits(df_dev, df_dev_clip, p, q, exog_cols):
    # Recebe o conjunto de desenvolvimento e devolve os splits de treino e validacao
    X = build_features(df_dev_clip, p, q, exog_cols)
    valid = X.notna().all(axis=1)
    X = X[valid]
    dates = df_dev_clip.loc[valid, "date"]

    y_clip = df_dev_clip.loc[valid, TARGET]
    y_orig = df_dev.loc[valid, TARGET]

    train_mask = dates < SPLIT_DATE_VAL
    val_mask = dates >= SPLIT_DATE_VAL
    return {
        "X_train": X[train_mask],
        "X_val": X[val_mask],
        "y_train_clip": y_clip[train_mask],
        "y_train_orig": y_orig[train_mask],
        "y_val": y_orig[val_mask],
    }


def criar_modelo(metodo, alpha):
    # Cria o modelo desejado
    if metodo == "LS":
        return LinearRegression()
    if metodo == "Ridge":
        return Ridge(alpha=alpha)
    if metodo == "Lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    raise ValueError(f"Invalid method: {metodo}")


def avaliar_modelos(split, alphas_ridge, alphas_lasso):
    # Avalia o método Least Squares, Ridge e Lasso
    mu, sd = metricas_normalizar(split["X_train"])
    X_tr = (split["X_train"] - mu) / sd
    X_val = (split["X_val"] - mu) / sd

    configs = ([("LS", None)] + [("Ridge", a) for a in alphas_ridge] + [("Lasso", a) for a in alphas_lasso])

    linhas = []

    for metodo, alpha in configs:
        model = criar_modelo(metodo, alpha)
        model.fit(X_tr, split["y_train_clip"])
        m_train = metrics_calc(split["y_train_orig"], model.predict(X_tr), Print=False)
        m_val = metrics_calc(split["y_val"], model.predict(X_val), Print=False)
        linhas.append({"metodo": metodo, "alpha": alpha, "rmse_train": m_train["rmse"], 
                       "rmse_val": m_val["rmse"], "r2_val": m_val["r2"]})
    return linhas

    
def grid_PQ(df_dev, df_dev_clip, exog_cols):
    # Cria uma database para comparar as várias iterações
    resultados = []
    for p in P_RANGE:
        for q in Q_RANGE:
            split = splits(df_dev, df_dev_clip, p, q, exog_cols)
            avaliacao = avaliar_modelos(split, ALPHAS_RIDGE, ALPHAS_LASSO)
            for linha in avaliacao:
                linha["P"] = p
                linha["Q"] = q #adiciona os indices de p e q à database
                resultados.append(linha)
    return pd.DataFrame(resultados)

def retreinar(df, p, q, metodo, alpha, cutoff, exog_cols):
    # Serve para reaproveitar o validation set antes de testar o model no test set - sem reescrever os hiperparametros
    train_days = df["date"] < cutoff
    limiar = df.loc[train_days, TARGET].quantile(CLIP_QUANTILE)
    df_clip = clip_target(df, limiar) # Cria uma nova df com a parte de teste clipped
 
    X = build_features(df_clip, p, q, exog_cols)
    valid = X.notna().all(axis=1)
    train_mask = valid & train_days
 
    mu, sd = metricas_normalizar(X[train_mask])
    model = criar_modelo(metodo, alpha)
    model.fit((X[train_mask] - mu) / sd, df_clip.loc[train_mask, TARGET])
    return {"model": model, "mu": mu, "sd": sd, "limiar": limiar, "X": X, "valid": valid}

def guardar_modelo(estado, caminho):
    # Serve para guardar o pickle
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(estado, f)


def main():
    # Bloco necessário para abrir o xlsx de qualquer pasta
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    # Criar a database
    df = load_data(args.data)
    exog_cols = list(df.columns[2:])  # coluna 0 = date, coluna 1 = clorofila, o resto sao as 12 exogenas

    # separa-se desde já os dados de teste, de treino e de validacao. O teste so e usado no final, para avaliar o modelo final
    df_dev = df[df["date"] < SPLIT_DATE_TEST].reset_index(drop=True)  

    # Fazer clip aos outliers (do treino) na database, antes de aplicar os lags
    limiar = df_dev.loc[df_dev["date"] < SPLIT_DATE_VAL, TARGET].quantile(CLIP_QUANTILE)
    print(f"Limiar do clipping (P{CLIP_QUANTILE}) = {limiar:.4f} mg/m3")
    df_dev_clip = clip_target(df_dev, limiar)

    # Gerar uma database com todos os dados acerca dos modelos 
    tabela = grid_PQ(df_dev, df_dev_clip, exog_cols)
    tabela = tabela.sort_values("rmse_val").reset_index(drop=True)

    # Melhores resultados
    print(tabela.head(10).to_string())
    print("\nMelhor LS:")
    print(tabela[tabela["metodo"] == "LS"].head(1).to_string())
    print("\nMelhor Ridge:")
    print(tabela[tabela["metodo"] == "Ridge"].head(1).to_string())
    print("\nMelhor Lasso:")
    print(tabela[tabela["metodo"] == "Lasso"].head(1).to_string())

    # Dá print ao melhor modelo com o melhor rmse
    best = tabela.iloc[0] # A melhor linha está na posicao 0
    p, q = int(best["P"]), int(best["Q"])
    metodo, alpha = best["metodo"], best["alpha"]
    print(f"\nEscolhido: {metodo} P={p} Q={q} alpha={alpha}")

    # Com os dados adquiridos volta a criar-se o modelo com os dados de treino e validação - sem alterar P, Q, alpha e o método escolhidos
    treino = retreinar(df, p, q, metodo, alpha, SPLIT_DATE_TEST, exog_cols)
    test_mask = treino["valid"] & (df["date"] >= SPLIT_DATE_TEST)
    X_test = (treino["X"][test_mask] - treino["mu"]) / treino["sd"]

    y_test = df.loc[test_mask, TARGET]
    y_test_pred = treino["model"].predict(X_test)

    metrics_calc(y_test, y_test_pred, label="Metricas do teste: \n\t", Print=True)

    # MODELO COM TODOS OS DADOS DO DATA SET -> para ser aplicado no test set final
    cutoff_final = df["date"].max() + pd.Timedelta(days=1)
    treino_final = retreinar(df, p, q, metodo, alpha, cutoff_final, exog_cols)
    cols = list(treino_final["X"].columns)
    mu = treino_final["mu"][cols].to_numpy()
    sd = treino_final["sd"][cols].to_numpy()
    limiar = float(treino_final["limiar"])
    modelo = treino_final["model"]

    #Nome das variáveis que são importadas pelo pickle
    estado = {
    "model": modelo, #Lasso
    "mu": mu, # um array com as médias de todas as colunas de X
    "sd": sd, # um array com as standart deviations de todas as colunas de X
    "limiar": limiar, # Valor definido para o clipping
    "P": p, # delay da clorofila
    "Q": q, # delay de todas as variáveis exogenas
    "feature_cols": cols, # a lista de colunas de X
    "exog_cols": exog_cols, # 12 variáveis exógenas na database (exceto data e clorofila)
    }

    # Gravar o estado no pkl e verificar
    guardar_modelo(estado, MODEL_PATH)
    (modelo.coef_ != 0).sum()
    print(f"Modelo guardado em {MODEL_PATH} ({(modelo.coef_ != 0).sum()} features ativas)")

if __name__ == "__main__":
    main()