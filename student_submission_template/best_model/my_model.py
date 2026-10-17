"""Final model: lagged linear regression (Lasso) with explicit clipping and normalisation."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from aux import preparar_features, TARGET


def load_model(model_path: str | Path) -> Any:
    """Load and return the fitted model and its plain numerical state."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the one-day-ahead chlorophyll-a prediction as an array with shape (1,)."""
    estado = model  # dicionario guardado no pkl; a chave "model" e o estimador

    # Lags, clipping do historico e normalizacao estao em aux.py (iguais ao treino)
    X_norm = preparar_features(history_df, day_features_df, estado)

    # DataFrame com os nomes das colunas: o sklearn confirma que a ordem bate com a do treino
    X_df = pd.DataFrame(X_norm, columns=estado["feature_cols"])
    pred = float(estado["model"].predict(X_df)[0])

    # Plano B (nao deve disparar): ultimo valor de clorofila do historico, ja limitado ao limiar
    if not np.isfinite(pred):
        ultimo = float(history_df.sort_values("date")[TARGET].iloc[-1])
        pred = min(ultimo, estado["limiar"])
    # Ultimo recurso: o intercepto (media da clorofila no treino)
    if not np.isfinite(pred):
        pred = float(estado["model"].intercept_)

    # Pos-processamento explicito: a concentracao nunca e negativa
    pred = max(0.0, pred)
    return np.asarray([pred], dtype=float)