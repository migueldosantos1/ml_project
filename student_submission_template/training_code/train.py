import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression, Ridge

DATA_PATH = "data/chlorophyll_student_2015_2023.xlsx"
TARGET = "chlorophyll_a_mg_m3"

# Corte cronologico: treino antes desta data, validacao a partir dela (nunca aleatorio)
SPLIT_DATE_VAL = "2020-01-01"
SPLIT_DATE_TEST = "2022-01-01"

# Limiar do clipping: percentil 99.9 do treino. Escolhido porque apanha os 3 erros de
# medicao (9.63, 6.93, 6.07) sem tocar nos blooms reais (maximo ~4.8)
CLIP_QUANTILE = 0.999

# Flags de diagnostico, so ligar quando for preciso ver
SHOW_BASELINE = False
SHOW_QUANTILES = False
SHOW_PEAKS = False

# Ranges para iteracao dos modelos
P_RANGE = [1, 3, 5, 7, 9]
Q_RANGE = [0, 1, 3, 5, 7, 9]
ALPHAS = [0.1, 1, 10, 100]

# Nome dos ficheiros para gravar a grid de iteração dos P, Q e alphas do ridge
GRID_CSV = "grid_resultados.csv"


def load_data(path):
    df = pd.read_excel(path, sheet_name="daily_data")
    df["date"] = pd.to_datetime(df["date"])  # strings -> datetime do pandas
    # Ordem cronologica obrigatoria: o shift(k) assume que cada linha e o dia a seguir a anterior
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df, P, Q, exog_cols):
    # Copia para nao alterar o df original (o predict tambem nao pode mexer no historico)
    copy_df = df.copy()
    # Dicionario {nome da coluna: coluna}. Se fizesse feats = ... dentro do ciclo, so
    # ficava com o ultimo lag, porque a variavel era sobrescrita a cada volta
    feats = {}

    # Lags da clorofila comecam em 1: o lag 0 seria o valor que queremos prever
    for i in range(1, P + 1):
        feats[f"chl_lag_{i}"] = copy_df[TARGET].shift(i)

    # Lags das exogenas comecam em 0: a medicao do proprio dia ja esta disponivel
    for col in exog_cols:
        for j in range(0, Q + 1):
            feats[f"{col}_lag_{j}"] = copy_df[col].shift(j)

    # Sazonalidade: o ano e um circulo, por isso o dia do ano vai para (sin, cos).
    # Assim 31 dez e 1 jan ficam proximos. Sao precisos os dois para cada dia ter um par unico
    doy = copy_df["date"].dt.dayofyear
    feats["s_year"] = np.sin(2 * np.pi * doy / 365.25)
    feats["c_year"] = np.cos(2 * np.pi * doy / 365.25)

    # Ordem das colunas = ordem de insercao no dicionario. Tem de ser igual no predict
    return pd.DataFrame(feats, index=copy_df.index)


def clip_target(df, upper):
    # Em vez de apagar linhas (o que desalinhava os lags), limitamos o valor maximo
    df_clip = df.copy()
    df_clip[TARGET] = df_clip[TARGET].clip(upper=upper)
    return df_clip


def metrics_calc(y_true, y_pred, label="", Print=True):
    # Ordem dos argumentos e sempre (verdadeiro, previsto)
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE em mg/m3, que e a metrica pedida no enunciado
    mae = metrics.mean_absolute_error(y_true, y_pred)
    if Print:
        print(f">>> {label} R2 - {r2:.4f}, MSE - {mse:.4f}, RMSE - {rmse:.4f}, MAE - {mae:.4f}")
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


def make_splits(df_dev, df_dev_clip, p, q, exog_cols):
    # Recebe so o conjunto de DESENVOLVIMENTO (treino + validacao). Os dias de teste nao
    # existem aqui dentro, por isso e impossivel que influenciem a escolha.
    # valid, y_clip e y_orig dependem de (p, q), logo recalculam-se a cada combinacao
    X = build_features(df_dev_clip, p, q, exog_cols)
    valid = X.notna().all(axis=1)
    X = X[valid]
    dates = df_dev.loc[valid, "date"]
 
    y_clip = df_dev_clip.loc[valid, TARGET]   # serve para treinar
    y_orig = df_dev.loc[valid, TARGET]        # serve para avaliar
 
    train_mask = dates < SPLIT_DATE_VAL
    val_mask = ~train_mask
    return {
        "X_train": X[train_mask], "X_val": X[val_mask],
        "y_train_clip": y_clip[train_mask], "y_train_orig": y_orig[train_mask],
        "y_val": y_orig[val_mask],
    }



def show_peaks(y_train, dates_train, threshold=4):
    # Para cada pico mostra o dia anterior e o seguinte.
    # Vizinhos normais e pico de 1 dia -> erro de medicao. Vizinhos tambem altos -> bloom real
    tabela = pd.DataFrame({
        "data": dates_train,
        "anterior": y_train.shift(1),
        "pico": y_train,
        "posterior": y_train.shift(-1),
    })
    # y_train > threshold e uma mascara booleana: so mostra as linhas True
    print(tabela[y_train > threshold])

def least_squares(X_train, y_train_clip, y_train_orig, X_val, y_val, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train_clip)
    print("LEAST SQUARES")
    metrics_calc(y_train_orig, model.predict(X_train), "Treino")
    metrics_calc(y_val, model.predict(X_val), "Validacao")
    metrics_calc(y_test, model.predict(X_test), "Teste")

def metricas_normalizar(X_train,):
    mu = X_train.mean()
    sd = X_train.std()
    sd = sd.replace(0, 1) #Se sd = 0 divisão daria NaN
    return mu, sd

def teste_ridge(alphas, X_train_norm, y_train_clip, y_train_orig, X_val_norm, y_val, X_test_norm, y_test):
    print("RIDGE")
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_norm, y_train_clip)                                    
        metrics_calc(y_train_orig, model.predict(X_train_norm), f"Treino (alpha={alpha})")   
        metrics_calc(y_val, model.predict(X_val_norm), f"Validacao (alpha={alpha})")
        metrics_calc(y_test, model.predict(X_test_norm), f"Teste (alpha={alpha})")


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
    if metodo == "LS":
        return LinearRegression()
    return Ridge(alpha=alpha)

def avaliar_modelos(split, alphas):
    # AINDA PODE SER ATUALIZADA COM MAIS MÉTODOS
    # Até agora o que faz é avaliar o método de Least Squares e o de Ridge
    mu, sd = metricas_normalizar(split["X_train"])
    X_tr = (split["X_train"] - mu) / sd
    X_val = (split["X_val"] - mu)/ sd

    for alpha in alphas:
        configs = [("LS", None)] + [("Ridge", alpha)]

    linhas = []

    for metodo, alpha in configs:
        model = criar_modelo(metodo, alpha)
        model.fit(X_tr, split["y_train_clip"])
        m_train = metrics_calc(split["y_train_orig"], model.predict(split["X_train"]), verbose = False)
        m_val = metrics_calc(split["y_val"], model.predict(split["X_val"]), verbose = False)
        linhas.append({"metodo": metodo, "alpha": alpha, "rmse_train": m_train["rmse"], 
                       "rmse_val": m_val["rmse"], "r2_val": m_val["r2"]})
    return linhas
    
def grid_PQ(df_dev, df_dev_clip, exog_cols):
    resultados = []
    for p in P_RANGE:
        for q in Q_RANGE:
            split = splits(df_dev, df_dev_clip, p, q, exog_cols)
            avaliacao = avaliar_modelos(split, ALPHAS)
            for linha in avaliacao:
                linha["P"] = p
                linha["Q"] = q #adiciona os indices de p e q à database
                resultados.append(linha)
    return pd.DataFrame(resultados)

def main():
    df = load_data(DATA_PATH)
    exog_cols = list(df.columns[2:])  # coluna 0 = date, coluna 1 = clorofila, o resto sao as 12 exogenas

    # separa-se desde já os dados de teste, de treino e de validacao. O teste so e usado no final, para avaliar o modelo final
    df_dev = df[df["date"] < SPLIT_DATE_TEST].reset_index(drop=True)  

    # Fazer clip aos outliers (do treino )na database, antes de aplicar os lags
    limiar = df_dev[df_dev["date"] < SPLIT_DATE_VAL, TARGET].quantile(CLIP_QUANTILE)
    print(f"Limiar do clipping (P{CLIP_QUANTILE}) = {limiar:.4f} mg/m3")
    df_dev_clip = clip_target(df_dev, limiar)

    # Gerar uma tabela com todos os dados acerca dos modelos 
    tabela = grid_PQ(df_dev, df_dev_clip, exog_cols)
    tabela = tabela.sort_values("rmse_val").reset_index(drop=True)
    tabela.to_csv(GRID_CSV, index=False)

    # Melhores resultados
    print(tabela.head(10).to_string())
    print("\nMelhor OLS:")
    print(tabela[tabela["metodo"] == "OLS"].head(1).to_string())
    print("Melhor Ridge:")
    print(tabela[tabela["metodo"] == "Ridge"].head(1).to_string())

    # Dá print ao melhor modelo com o melhor rmse
    best = tabela.iloc[0] # A melhor linha está na posicao 0
    p, q = int(best["P"]), int(best["Q"])
    metodo, alpha = best["metodo"], best["alpha"]
    print(f"\nEscolhido: {metodo} P={p} Q={q} alpha={alpha}")

if __name__ == "__main__":
    main()
