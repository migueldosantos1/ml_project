import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression, Ridge

DATA_PATH = "data/chlorophyll_student_2015_2023.xlsx"
TARGET = "chlorophyll_a_mg_m3"

# Tres blocos cronologicos: treino (antes de VAL), validacao (escolher hiperparametros),
# teste interno (so se olha no fim, uma vez, para a configuracao escolhida)
SPLIT_DATE_VAL = "2020-01-01"
SPLIT_DATE_TEST = "2022-01-01"

# Limiar do clipping: percentil 99.9 do treino
CLIP_QUANTILE = 0.999

# Listas (nao conjuntos): a ordem fica garantida e e mais facil de ler na tabela final
P_RANGE = [1, 3, 5, 7, 9]
Q_RANGE = [0, 1, 3, 5, 7, 9]
ALPHAS = [0.1, 1, 10, 100]

SHOW_BASELINE = False


def load_data(path):
    df = pd.read_excel(path, sheet_name="daily_data")
    df["date"] = pd.to_datetime(df["date"])
    # Ordem cronologica obrigatoria: o shift(k) assume que cada linha e o dia a seguir a anterior
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df, P, Q, exog_cols):
    copy_df = df.copy()
    feats = {}

    # Lags da clorofila comecam em 1: o lag 0 seria o valor que queremos prever
    for i in range(1, P + 1):
        feats[f"chl_lag_{i}"] = copy_df[TARGET].shift(i)

    # Lags das exogenas comecam em 0: a medicao do proprio dia ja esta disponivel
    for col in exog_cols:
        for j in range(0, Q + 1):
            feats[f"{col}_lag_{j}"] = copy_df[col].shift(j)

    # Sazonalidade: o ano e um circulo, por isso o dia do ano vai para (sin, cos)
    doy = copy_df["date"].dt.dayofyear
    feats["s_year"] = np.sin(2 * np.pi * doy / 365.25)
    feats["c_year"] = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame(feats, index=copy_df.index)


def clip_target(df, upper):
    # Em vez de apagar linhas (o que desalinhava os lags), limitamos o valor maximo
    df_clip = df.copy()
    df_clip[TARGET] = df_clip[TARGET].clip(upper=upper)
    return df_clip


def metrics_calc(y_true, y_pred, label="", verbose=True):
    # Ordem dos argumentos e sempre (verdadeiro, previsto)
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    if verbose:
        print(f">>> {label} R2 - {r2:.4f}, MSE - {mse:.4f}, RMSE - {rmse:.4f}, MAE - {mae:.4f}")
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


def make_splits(df, df_clip, p, q, exog_cols):
    # Tudo o que depende de (p, q) fica aqui dentro: as features, as linhas validas e os targets.
    # O numero de linhas validas muda com p e q (lags maiores => mais linhas iniciais com NaN),
    # por isso valid, y_clip e y_orig tem de ser recalculados para cada combinacao
    X = build_features(df_clip, p, q, exog_cols)
    valid = X.notna().all(axis=1)
    X = X[valid]
    dates = df.loc[valid, "date"]

    # Dois targets: o cortado serve para treinar, o original serve para avaliar
    y_clip = df_clip.loc[valid, TARGET]
    y_orig = df.loc[valid, TARGET]

    train_mask = dates < SPLIT_DATE_VAL
    val_mask = (dates >= SPLIT_DATE_VAL) & (dates < SPLIT_DATE_TEST)
    test_mask = dates >= SPLIT_DATE_TEST

    return {
        "X_train": X[train_mask], "X_val": X[val_mask], "X_test": X[test_mask],
        "y_train_clip": y_clip[train_mask], "y_train_orig": y_orig[train_mask],
        "y_val": y_orig[val_mask], "y_test": y_orig[test_mask],
    }


def normalizar(X_train, X_val, X_test):
    # mu e sd saem SO do treino e aplicam-se aos tres conjuntos
    mu = X_train.mean()
    sd = X_train.std()
    sd = sd.replace(0, 1)  # coluna constante: dividir por 0 daria NaN
    return (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd, mu, sd


def avaliar_modelos(s, alphas):
    # Treina OLS e Ridge (varios alpha) sobre os mesmos dados normalizados.
    # O OLS e invariante a escala, por isso normalizar nao o altera.
    # Devolve uma linha por modelo, com treino e validacao (o teste nao entra aqui)
    X_tr, X_va, X_te, mu, sd = normalizar(s["X_train"], s["X_val"], s["X_test"])

    modelos = [("OLS", None, LinearRegression())]
    for a in alphas:
        modelos.append(("Ridge", a, Ridge(alpha=a)))

    linhas = []
    for nome, alpha, model in modelos:
        model.fit(X_tr, s["y_train_clip"])
        m_train = metrics_calc(s["y_train_orig"], model.predict(X_tr), verbose=False)
        m_val = metrics_calc(s["y_val"], model.predict(X_va), verbose=False)
        linhas.append({"metodo": nome, "alpha": alpha,
                       "rmse_train": m_train["rmse"],
                       "rmse_val": m_val["rmse"], "r2_val": m_val["r2"]})
    return linhas


def grid_PQ(df, df_clip, exog_cols):
    resultados = []
    for p in P_RANGE:
        for q in Q_RANGE:
            s = make_splits(df, df_clip, p, q, exog_cols)
            for linha in avaliar_modelos(s, ALPHAS):
                linha["P"] = p
                linha["Q"] = q
                resultados.append(linha)
    return pd.DataFrame(resultados)


def main():
    df = load_data(DATA_PATH)
    exog_cols = list(df.columns[2:])  # coluna 0 = date, coluna 1 = clorofila, o resto sao as 12 exogenas

    # Limiar aprendido so com o treino, usando a clorofila ORIGINAL
    train_days = df["date"] < SPLIT_DATE_VAL
    limiar = df.loc[train_days, TARGET].quantile(CLIP_QUANTILE)
    print(f"Limiar de clipping (P{CLIP_QUANTILE * 100:g}): {limiar:.4f}")

    # Cortamos o df ANTES de criar os lags, para os lags ja herdarem o valor cortado
    df_clip = clip_target(df, limiar)

    if SHOW_BASELINE:
        s = make_splits(df, df_clip, 3, 7, exog_cols)
        metrics_calc(s["y_val"], s["X_val"]["chl_lag_1"], "Persistencia (val)")

    # Grid: cada linha da tabela e um (P, Q, metodo, alpha). Escolhe-se pelo erro de VALIDACAO
    tabela = grid_PQ(df, df_clip, exog_cols)
    tabela = tabela.sort_values("rmse_val").reset_index(drop=True)
    print(tabela.head(15).to_string())

    # Melhor OLS e melhor Ridge, para a tabela do relatorio
    print("\nMelhor OLS:")
    print(tabela[tabela["metodo"] == "OLS"].head(1).to_string())
    print("Melhor Ridge:")
    print(tabela[tabela["metodo"] == "Ridge"].head(1).to_string())

    # O teste interno so se olha agora, uma vez, para a configuracao vencedora
    best = tabela.iloc[0]
    s = make_splits(df, df_clip, int(best["P"]), int(best["Q"]), exog_cols)
    X_tr, X_va, X_te, mu, sd = normalizar(s["X_train"], s["X_val"], s["X_test"])
    if best["metodo"] == "OLS":
        model = LinearRegression()
    else:
        model = Ridge(alpha=best["alpha"])
    model.fit(X_tr, s["y_train_clip"])
    print(f"\nConfiguracao escolhida: {best['metodo']} P={int(best['P'])} Q={int(best['Q'])} alpha={best['alpha']}")
    metrics_calc(s["y_val"], model.predict(X_va), "Validacao")
    metrics_calc(s["y_test"], model.predict(X_te), "Teste interno")


if __name__ == "__main__":
    main()