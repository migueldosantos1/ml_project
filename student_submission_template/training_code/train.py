import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression, Ridge

DATA_PATH = "data/chlorophyll_student_2015_2023.xlsx"
TARGET = "chlorophyll_a_mg_m3"

# P = nº de lags da clorofila (1..P), Q = nº de lags das v. exogenas (0..Q)
# Sao hiperparametros: a escolha final vem do erro de validacao (grid search mais tarde)
P = 3
Q = 7

# Corte cronologico: treino antes desta data, validacao a partir dela (nunca aleatorio)
SPLIT_DATE = "2022-01-01"

# Limiar do clipping: percentil 99.9 do treino. Escolhido porque apanha os 3 erros de
# medicao (9.63, 6.93, 6.07) sem tocar nos blooms reais (maximo ~4.8)
CLIP_QUANTILE = 0.999

# Flags de diagnostico, so ligar quando for preciso ver
SHOW_BASELINE = False
SHOW_QUANTILES = False
SHOW_PEAKS = False


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


def metrics_calc(y_true, y_pred, label=""):
    # Ordem dos argumentos e sempre (verdadeiro, previsto)
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE em mg/m3, que e a metrica pedida no enunciado
    mae = metrics.mean_absolute_error(y_true, y_pred)
    print(f">>> {label} R2 - {r2:.4f}, MSE - {mse:.4f}, RMSE - {rmse:.4f}, MAE - {mae:.4f}")
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


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

def normalizar(X_train, X_val):
    mu = X_train.mean()
    sd = X_train.std()
    sd = sd.replace(0, 1) #Se sd = 0 divisão daria NaN
    X_train_norm = (X_train - mu) / sd
    X_val_norm = (X_val - mu) / sd
    return X_train_norm, X_val_norm, mu, sd

def teste_ridge(alphas, X_train_norm, y_train_clip, y_train_orig, X_val_norm, y_val):
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_norm, y_train_clip)                                    
        m_train = metrics_calc(y_train_orig, model.predict(X_train_norm), f"Treino (alpha={alpha})")   
        m_val = metrics_calc(y_val, model.predict(X_val_norm), f"Validacao (alpha={alpha})")

def main():
    df = load_data(DATA_PATH)
    exog_cols = list(df.columns[2:])  # coluna 0 = date, coluna 1 = clorofila, o resto sao as 12 exogenas

    # O limiar tem de ser aprendido so com o treino, senao a validacao influenciava um parametro.
    # Usa-se a clorofila ORIGINAL (antes de cortar) para o quantil
    train_days = df["date"] < SPLIT_DATE
    limiar = df.loc[train_days, TARGET].quantile(CLIP_QUANTILE)
    print(f"Limiar de clipping (P{CLIP_QUANTILE * 100:g}): {limiar:.4f}")

    # Cortamos o df ANTES de criar os lags, para os lags da clorofila ja herdarem o valor cortado
    # (um pico aparece como target em t, mas tambem como lag 1, 2 e 3 nos dias seguintes)
    df_clip = clip_target(df, limiar)
    X = build_features(df_clip, P, Q, exog_cols)

    # Linhas com NaN sao as primeiras (lags ainda sem historico). A regressao nao aceita NaN.
    # .all(axis=1) pergunta, linha a linha, se todas as features sao validas
    valid = X.notna().all(axis=1)
    X = X[valid]
    dates = df.loc[valid, "date"]  # mesma mascara em X, y e dates para ficarem sempre alinhados

    # Dois targets: o cortado serve para treinar (nao deformar os coeficientes),
    # o original serve para avaliar (2024 tambem vai ter outliers, por isso nao se limpa a validacao)
    y_clip = df_clip.loc[valid, TARGET]
    y_orig = df.loc[valid, TARGET]

    # ~train_mask inverte a mascara: True passa a False, ou seja, fica a validacao
    train_mask = dates < SPLIT_DATE
    X_train, X_val = X[train_mask], X[~train_mask]
    y_train_clip = y_clip[train_mask]
    y_train_orig = y_orig[train_mask]
    y_val = y_orig[~train_mask]
    print(X_train.shape, X_val.shape)

    if SHOW_BASELINE:
        # Persistencia: prever y(t) = y(t-1). Nao tem parametros, e so uma regra fixa de referencia
        metrics_calc(y_val, X_val["chl_lag_1"], "Persistencia (val)")
    if SHOW_QUANTILES:
        print(y_train_orig.quantile([0.95, 0.99, 0.999]))
        print("Maximo:", y_train_orig.max())
    if SHOW_PEAKS:
        show_peaks(y_train_orig, dates[train_mask])

    # fit estima os pesos (theta) e o intercepto minimizando a soma dos erros ao quadrado.
    # Nao se adiciona coluna de uns: o LinearRegression ja estima o intercepto
    model = LinearRegression().fit(X_train, y_train_clip)

    # Treino vs validacao: se o treino for muito melhor, ha overfitting (variancia alta)
    metrics_calc(y_train_orig, model.predict(X_train), "Treino")
    metrics_calc(y_val, model.predict(X_val), "Validacao")

    #Normalizar os dados
    X_train_norm, X_val_norm, mu, sd = normalizar(X_train, X_val)

    # Testar varios alphas para o Ridge
    alphas = [0.1, 1, 10]
    teste_ridge(alphas, X_train_norm, y_train_clip, y_train_orig, X_val_norm, y_val)


if __name__ == "__main__":
    main()