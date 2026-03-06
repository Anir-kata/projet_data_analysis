import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.logging_utils import get_logger

logger = get_logger(__name__)


def fit_arima(series: pd.Series, order: tuple = (1, 1, 1)):
    """Fit an ARIMA model on a univariate time series.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by date or numeric period.
    order : tuple
        ARIMA order (p,d,q).

    Returns
    -------
    statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA model instance.
    """
    logger.info(f"Fitting ARIMA{order} model on series of length {len(series)}")
    try:
        model = sm.tsa.ARIMA(series, order=order)
        fitted = model.fit()
        logger.info("ARIMA fit completed")
        return fitted
    except Exception as e:
        logger.error(f"Failed to fit ARIMA model: {e}")
        raise


def forecast_arima(fitted_model, steps: int) -> pd.DataFrame:
    """Produce a forecast dataframe for a fitted ARIMA model.

    Parameters
    ----------
    fitted_model : statsmodels ARIMAResults
        Model previously trained with `fit_arima`.
    steps : int
        Number of future periods to forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame containing predictions along with confidence intervals.
        Columns:
            - period
            - mean
            - mean_ci_lower
            - mean_ci_upper
    """
    logger.info(f"Forecasting {steps} periods ahead")
    try:
        pred = fitted_model.get_forecast(steps=steps)
        df_forecast = pred.summary_frame().reset_index()
        return df_forecast
    except Exception as e:
        logger.error(f"Failed to forecast ARIMA: {e}")
        raise
    df_forecast = df_forecast.rename(columns={"index": "period"})
    return df_forecast
