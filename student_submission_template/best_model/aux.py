"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

TARGET = "chlorophyll_a_mg_m3"

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

def preparar_features(history_df, day_features_df, estado):
    history = history_df.copy()
    day = day_features_df.copy()

    history["date"] = pd.to_datetime(history["date"]) #Colocar em formato de data
    day["date"] = pd.to_datetime(day["date"])

    history = history.sort_values("date").reset_index(drop=True) #Ordenar
    history[TARGET] = history[TARGET].clip(upper=estado["limiar"]) #Clip
    history = history.tail(max(estado["P"], estado["Q"]) + 1) #grava em history apenas as ultimas max(estado["P"], estado["Q"]) + 1 linhas

    day[TARGET] = np.nan
    day = day[history.columns]

    total = pd.concat([history, day], ignore_index=True)

    X = build_features(total, estado["P"], estado["Q"], estado["exog_cols"])
    X = X.iloc[[-1]] # um DataFrame com uma linha -> 2D, shape (1, 77)
    X = X[estado["feature_cols"]]

    X_norm = (X.to_numpy() - estado["mu"]) / estado["sd"]
    return X_norm
