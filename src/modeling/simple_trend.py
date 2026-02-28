import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from src.logging_utils import get_logger

logger = get_logger(__name__)

def fit_linear_trend(df_yearly: pd.DataFrame, target_col: str) -> LinearRegression:
    """
    Ajuste une régression linéaire simple sur (annee, target).
    """
    logger.info(f"Fitting linear trend for {target_col}")

    X = df_yearly[["annee"]].values
    y = df_yearly[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    logger.info(
        f"Fitted model: y = {model.coef_[0]:.2f} * year + {model.intercept_:.2f}"
    )
    return model

def predict_future(model: LinearRegression, years: list[int]) -> pd.DataFrame:
    """
    Prédit la variable cible pour une liste d'années futures.
    """
    X_future = np.array(years).reshape(-1, 1)
    y_pred = model.predict(X_future)
    return pd.DataFrame({"annee": years, "prediction": y_pred})