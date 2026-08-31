from __future__ import annotations

import json
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parents[1]

# January 2022 is used only for model/configuration selection.
SELECTION_START = pd.Timestamp("2022-01-01")
SELECTION_END = pd.Timestamp("2022-01-31")

# February 2022 is the final blind forecast shown by the website.
FINAL_FORECAST_START = pd.Timestamp("2022-02-01")
FINAL_FORECAST_END = pd.Timestamp("2022-02-28")

MIN_TRAIN_ROWS = 60
MAX_VAR_LAGS = 10
CONFIDENCE_ALPHA = 0.05
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
VAR_MC_SIMS = int(os.getenv("VAR_MC_SIMS", "1000"))
VAR_MC_SEED = int(os.getenv("VAR_MC_SEED", "0"))
SENTIMENT_LAGS = [1, 2, 3]

TARGET_ASSETS = [
    "DSEX",
    "DS30",
    "DSES",
    "GP",
    "ACI",
    "BEXIMCO",
    "BRACBANK",
    "BXPHARMA",
]

TICKER_ALIASES = {
    "DSEX": ["DSEX", "00DSEX"],
    "DS30": ["DS30", "00DS30"],
    "DSES": ["DSES", "00DSES"],
    "GP": ["GP"],
    "ACI": ["ACI"],
    "BEXIMCO": ["BEXIMCO"],
    "BRACBANK": ["BRACBANK"],
    "BXPHARMA": ["BXPHARMA"],
}

EXPERIMENTS = [
    (None, "Baseline"),
    (["lex_title"], "Lexicon-Title"),
    (["lex_content"], "Lexicon-Content"),
    (["lex_title", "lex_content"], "Lexicon-Title+Content"),
    (["bangla_title"], "BanglaBERT-Title"),
    (["bangla_content"], "BanglaBERT-Content"),
    (["bangla_title", "bangla_content"], "BanglaBERT-Title+Content"),
]

MODEL_NAMES = ["Prophet", "SARIMAX", "VAR"]
SENTIMENT_COLUMNS = ["bangla_title", "bangla_content", "lex_title", "lex_content"]

# Known DSE holiday in the final blind month. Fridays and Saturdays are removed separately.
DEFAULT_FINAL_HOLIDAYS = {pd.Timestamp("2022-02-21")}


@dataclass
class PipelinePaths:
    price_directory: Path
    banglabert_file: Path
    lexicon_file: Path
    output_directory: Path
    app_data_directory: Path


def resolve_paths() -> PipelinePaths:
    local_input = APP_DIR / "input_data"

    kaggle_price = Path("/kaggle/input/datasets/mdaltaf2k2/dse-unadjusted-data")
    kaggle_bangla = Path(
        "/kaggle/input/datasets/mdaltaf2k2/prothom-alo-sentiment-score/"
        "daily_sentiment_banglabert.csv"
    )
    kaggle_lexicon = Path(
        "/kaggle/input/datasets/mdaltaf2k2/prothom-alo-sentiment-score/"
        "daily_sentiment_lexicon.csv"
    )

    def env_or_default(name: str, *candidates: Path) -> Path:
        value = os.getenv(name)
        if value:
            return Path(value).expanduser().resolve()
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[-1].resolve()

    price_directory = env_or_default(
        "DSE_PRICE_DIR", kaggle_price, local_input / "dse-unadjusted-data"
    )
    banglabert_file = env_or_default(
        "BANGLABERT_FILE", kaggle_bangla, local_input / "daily_sentiment_banglabert.csv"
    )
    lexicon_file = env_or_default(
        "LEXICON_FILE", kaggle_lexicon, local_input / "daily_sentiment_lexicon.csv"
    )

    output_directory = Path(
        os.getenv("PIPELINE_OUTPUT_DIR", str(APP_DIR / "pipeline_output"))
    ).expanduser().resolve()
    app_data_directory = Path(
        os.getenv("FORECAST_DATA_DIR", str(APP_DIR / "data" / "app_data"))
    ).expanduser().resolve()

    return PipelinePaths(
        price_directory=price_directory,
        banglabert_file=banglabert_file,
        lexicon_file=lexicon_file,
        output_directory=output_directory,
        app_data_directory=app_data_directory,
    )


def find_column(columns: Iterable[str], candidates: Iterable[str]):
    column_map = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        matched = column_map.get(candidate.strip().lower())
        if matched is not None:
            return matched
    return None


def convert_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def classify_asset(name: str) -> str:
    return "Index" if name in {"DSEX", "DS30", "DSES"} else "Company"


def validate_input_paths(paths: PipelinePaths) -> None:
    if not paths.price_directory.exists():
        raise FileNotFoundError(
            f"DSE price directory not found: {paths.price_directory}. "
            "Set DSE_PRICE_DIR or copy files into input_data/dse-unadjusted-data."
        )
    if not paths.banglabert_file.exists():
        raise FileNotFoundError(
            f"BanglaBERT sentiment file not found: {paths.banglabert_file}. "
            "Set BANGLABERT_FILE or copy it into input_data."
        )
    if not paths.lexicon_file.exists():
        raise FileNotFoundError(
            f"Lexicon sentiment file not found: {paths.lexicon_file}. "
            "Set LEXICON_FILE or copy it into input_data."
        )


def prepare_sentiment(
    data: pd.DataFrame,
    title_column: str,
    content_column: str,
    prefix: str,
) -> pd.DataFrame:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ds"] = df["date"]
    df[f"{prefix}_title"] = pd.to_numeric(df[title_column], errors="coerce")
    df[f"{prefix}_content"] = pd.to_numeric(df[content_column], errors="coerce")
    return (
        df[["ds", f"{prefix}_title", f"{prefix}_content"]]
        .dropna(subset=["ds"])
        .groupby("ds", as_index=False)
        .mean()
        .sort_values("ds")
        .reset_index(drop=True)
    )


def load_sentiment(paths: PipelinePaths):
    bangla = pd.read_csv(paths.banglabert_file)
    lexicon = pd.read_csv(paths.lexicon_file)

    bangla_feat = prepare_sentiment(
        bangla, "title_score_mean", "content_score_mean", "bangla"
    )
    lexicon_feat = prepare_sentiment(
        lexicon, "title_score_mean", "content_score_mean", "lex"
    )
    return bangla_feat, lexicon_feat


def find_asset_files(price_directory: Path) -> dict[str, Path]:
    company_files = sorted(price_directory.glob("*_data.csv"))
    if not company_files:
        raise FileNotFoundError(f"No *_data.csv files found in {price_directory}")

    companies: dict[str, Path] = {}
    for target in TARGET_ASSETS:
        aliases = {x.upper() for x in TICKER_ALIASES[target]}
        found = None
        for filepath in company_files:
            ticker = filepath.name.replace("_data.csv", "").upper()
            if ticker in aliases:
                found = filepath
                break
        if found is not None:
            companies[target] = found

    missing = [asset for asset in TARGET_ASSETS if asset not in companies]
    if missing:
        raise FileNotFoundError(f"Missing selected DSE asset files: {missing}")
    return companies


def load_price_data(filepath: Path) -> pd.DataFrame:
    raw = pd.read_csv(filepath)
    date_col = find_column(raw.columns, ["Date", "Trading Date", "Trade Date"])
    close_col = find_column(raw.columns, ["Close", "Closing Price", "Close Price"])
    volume_col = find_column(
        raw.columns,
        ["Volume", "Trade Volume", "Total Volume", "Trading Volume", "Vol", "Vol."],
    )

    if date_col is None:
        raise ValueError(f"Date column not found in {filepath.name}")
    if close_col is None:
        raise ValueError(f"Close column not found in {filepath.name}")

    df = pd.DataFrame()
    df["ds"] = pd.to_datetime(raw[date_col], errors="coerce")
    df["y"] = convert_numeric(raw[close_col])
    df["volume"] = convert_numeric(raw[volume_col]) if volume_col is not None else np.nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ds", "y"])
    df = df[df["y"] > 0]
    df.loc[df["volume"] < 0, "volume"] = np.nan
    return (
        df.sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )


def prepare_full_data(
    price: pd.DataFrame,
    bangla_feat: pd.DataFrame,
    lexicon_feat: pd.DataFrame,
) -> pd.DataFrame:
    df = (
        price.merge(bangla_feat, on="ds", how="left")
        .merge(lexicon_feat, on="ds", how="left")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    # Only chronological forward fill is allowed. Future sentiment never moves backward.
    df[SENTIMENT_COLUMNS] = df[SENTIMENT_COLUMNS].ffill().fillna(0)
    return df


def add_sentiment_lags(df: pd.DataFrame, sentiment_columns: list[str]):
    data = df.copy()
    lag_columns: list[str] = []
    for col in sentiment_columns:
        for lag in SENTIMENT_LAGS:
            lag_name = f"{col}_lag{lag}"
            data[lag_name] = data[col].shift(lag)
            lag_columns.append(lag_name)
    return data, lag_columns


def get_frozen_sentiment_lags(
    full_data: pd.DataFrame,
    sentiment_columns: list[str],
    forecast_origin: pd.Timestamp,
) -> dict[str, float]:
    history = full_data[full_data["ds"] <= forecast_origin].copy()
    if len(history) < max(SENTIMENT_LAGS):
        raise ValueError("Not enough historical observations for sentiment lags")

    frozen: dict[str, float] = {}
    for col in sentiment_columns:
        values = history[col].astype(float).reset_index(drop=True)
        for lag in SENTIMENT_LAGS:
            frozen[f"{col}_lag{lag}"] = float(values.iloc[-lag])
    return frozen


def calculate_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        raise ValueError("No valid observations")

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else np.nan
    )
    nrmse = float(rmse / np.mean(np.abs(actual)) * 100)
    try:
        r2 = float(r2_score(actual, predicted))
    except Exception:
        r2 = np.nan

    if len(actual) > 1:
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)
    else:
        directional_accuracy = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "NRMSE (%)": nrmse,
        "R2": r2,
        "Directional Accuracy (%)": directional_accuracy,
    }


def observed_dates(
    full_data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    dates = full_data.loc[
        (full_data["ds"] >= start) & (full_data["ds"] <= end), "ds"
    ].drop_duplicates().sort_values()
    return dates.reset_index(drop=True)


def generate_dse_future_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    holidays: set[pd.Timestamp] | None = None,
) -> pd.Series:
    holidays = holidays or set()
    dates = []
    for day in pd.date_range(start, end, freq="D"):
        # Python weekday: Monday=0 ... Friday=4, Saturday=5, Sunday=6.
        if day.weekday() in {4, 5}:  # DSE weekend: Friday and Saturday
            continue
        if day.normalize() in {h.normalize() for h in holidays}:
            continue
        dates.append(day.normalize())
    return pd.Series(dates, dtype="datetime64[ns]")


def create_prophet() -> Prophet:
    return Prophet(
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_range=0.95,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        interval_width=1 - CONFIDENCE_ALPHA,
    )


def run_prophet(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    future_dates: pd.Series,
    sentiment_columns: list[str] | None,
) -> pd.DataFrame:
    if sentiment_columns is None:
        model = create_prophet()
        model.fit(train_data[["ds", "y"]])
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)
    else:
        lagged, lag_columns = add_sentiment_lags(full_data, sentiment_columns)
        train_lagged = lagged[lagged["ds"] <= forecast_origin].dropna(subset=lag_columns)
        if len(train_lagged) < MIN_TRAIN_ROWS:
            raise ValueError("Not enough Prophet training observations")

        model = create_prophet()
        for col in lag_columns:
            model.add_regressor(col)
        model.fit(train_lagged[["ds", "y"] + lag_columns])

        frozen = get_frozen_sentiment_lags(full_data, sentiment_columns, forecast_origin)
        future = pd.DataFrame({"ds": future_dates})
        for col in lag_columns:
            future[col] = frozen[col]
        forecast = model.predict(future)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates).values,
            "Predicted": forecast["yhat"].values,
            "Lower": forecast["yhat_lower"].values,
            "Upper": forecast["yhat_upper"].values,
        }
    )


def run_sarimax(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    future_dates: pd.Series,
    sentiment_columns: list[str] | None,
) -> pd.DataFrame:
    if sentiment_columns is None:
        model = sm.tsa.statespace.SARIMAX(
            train_data["y"].values,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.get_forecast(steps=len(future_dates))
    else:
        lagged, lag_columns = add_sentiment_lags(full_data, sentiment_columns)
        train_lagged = lagged[lagged["ds"] <= forecast_origin].dropna(subset=lag_columns)
        if len(train_lagged) < MIN_TRAIN_ROWS:
            raise ValueError("Not enough SARIMAX training observations")

        model = sm.tsa.statespace.SARIMAX(
            train_lagged["y"].values,
            exog=train_lagged[lag_columns].values,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        frozen = get_frozen_sentiment_lags(full_data, sentiment_columns, forecast_origin)
        future_exog = np.array(
            [[frozen[col] for col in lag_columns] for _ in range(len(future_dates))]
        )
        forecast = fitted.get_forecast(steps=len(future_dates), exog=future_exog)

    predicted = np.asarray(forecast.predicted_mean)
    ci = np.asarray(forecast.conf_int(alpha=CONFIDENCE_ALPHA))
    return pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates).values,
            "Predicted": predicted,
            "Lower": ci[:, 0],
            "Upper": ci[:, 1],
        }
    )


def prepare_var(full_data: pd.DataFrame, sentiment_columns: list[str] | None):
    columns = ["ds", "y", "volume"]
    if sentiment_columns is not None:
        columns += sentiment_columns
    data = full_data[columns].copy()

    data["y_ret"] = np.log(data["y"]).diff()
    data["volume_change"] = np.log1p(data["volume"]).diff()
    var_columns = ["y_ret", "volume_change"]

    if sentiment_columns is not None:
        for col in sentiment_columns:
            diff_col = f"{col}_diff"
            data[diff_col] = data[col].diff()
            var_columns.append(diff_col)

    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=var_columns)
    return data, var_columns


def choose_var_lag(train_var: pd.DataFrame):
    model = VAR(train_var)
    max_lags = min(MAX_VAR_LAGS, max(1, len(train_var) // 20))
    try:
        selection = model.select_order(maxlags=max_lags)
        lag = selection.selected_orders.get("aic", 1)
        if lag is None or lag < 1:
            lag = 1
    except Exception:
        lag = 1
    return model, int(lag)


def run_var(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    future_dates: pd.Series,
    sentiment_columns: list[str] | None,
    compute_intervals: bool = True,
) -> pd.DataFrame:
    if full_data["volume"].notna().sum() == 0:
        raise ValueError("VAR requires usable Volume data")

    var_data, var_columns = prepare_var(full_data, sentiment_columns)
    train_var = var_data[var_data["ds"] <= forecast_origin][var_columns].copy()
    if len(train_var) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough VAR training rows: {len(train_var)}")

    constant = [c for c in var_columns if np.isclose(train_var[c].std(), 0)]
    if constant:
        raise ValueError(f"Constant VAR variables: {constant}")

    model, lag_order = choose_var_lag(train_var)
    fitted = model.fit(lag_order, trend="c")
    if not fitted.is_stable():
        raise ValueError("VAR model is unstable")

    n_steps = len(future_dates)
    history = train_var.values[-lag_order:].copy()
    sentiment_indices: list[int] = []
    if sentiment_columns is not None:
        for col in sentiment_columns:
            sentiment_indices.append(var_columns.index(f"{col}_diff"))

    predicted_returns = []
    for _ in range(n_steps):
        next_value = fitted.intercept.copy()
        for lag in range(lag_order):
            next_value += fitted.coefs[lag] @ history[-(lag + 1)]
        # Future sentiment is frozen, therefore first-difference sentiment is zero.
        for idx in sentiment_indices:
            next_value[idx] = 0.0
        predicted_returns.append(next_value[var_columns.index("y_ret")])
        history = np.vstack([history[1:], next_value])

    predicted_returns = np.asarray(predicted_returns)
    last_price = float(train_data["y"].iloc[-1])
    predicted_price = last_price * np.exp(np.cumsum(predicted_returns))

    if compute_intervals:
        # Monte Carlo intervals are needed for the final public forecast only.
        # January model selection uses RMSE from point forecasts, so skipping
        # these simulations there greatly reduces free-cloud CPU time.
        rng = np.random.default_rng(VAR_MC_SEED)
        sigma = np.asarray(fitted.sigma_u)
        simulations = np.zeros((VAR_MC_SIMS, n_steps))

        for sim in range(VAR_MC_SIMS):
            hist = train_var.values[-lag_order:].copy()
            sim_returns = []
            for _ in range(n_steps):
                shock = rng.multivariate_normal(np.zeros(len(var_columns)), sigma)
                next_value = fitted.intercept.copy()
                for lag in range(lag_order):
                    next_value += fitted.coefs[lag] @ hist[-(lag + 1)]
                next_value += shock
                for idx in sentiment_indices:
                    next_value[idx] = 0.0
                sim_returns.append(next_value[var_columns.index("y_ret")])
                hist = np.vstack([hist[1:], next_value])
            simulations[sim, :] = last_price * np.exp(np.cumsum(sim_returns))

        lower = np.percentile(
            simulations, 100 * CONFIDENCE_ALPHA / 2, axis=0
        )
        upper = np.percentile(
            simulations, 100 * (1 - CONFIDENCE_ALPHA / 2), axis=0
        )
    else:
        lower = np.full(n_steps, np.nan)
        upper = np.full(n_steps, np.nan)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates).values,
            "Predicted": predicted_price,
            "Lower": lower,
            "Upper": upper,
        }
    )


def run_model(
    model_name: str,
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    future_dates: pd.Series,
    sentiment_columns: list[str] | None,
    compute_intervals: bool = True,
) -> pd.DataFrame:
    if model_name == "Prophet":
        return run_prophet(
            full_data, train_data, forecast_origin, future_dates, sentiment_columns
        )
    if model_name == "SARIMAX":
        return run_sarimax(
            full_data, train_data, forecast_origin, future_dates, sentiment_columns
        )
    if model_name == "VAR":
        return run_var(
            full_data,
            train_data,
            forecast_origin,
            future_dates,
            sentiment_columns,
            compute_intervals=compute_intervals,
        )
    raise ValueError(f"Unknown model: {model_name}")


def process_asset(
    asset: str,
    filepath: Path,
    bangla_feat: pd.DataFrame,
    lexicon_feat: pd.DataFrame,
    selection_forecast_directory: Path,
    final_forecast_directory: Path,
):
    print(f"\n[START] {asset}", flush=True)
    price = load_price_data(filepath)
    full_data = prepare_full_data(price, bangla_feat, lexicon_feat)

    # ------------------------------------------------------------
    # Stage 1: January 2022 model/configuration selection
    # ------------------------------------------------------------
    selection_training = full_data[full_data["ds"] < SELECTION_START].copy()
    if len(selection_training) < MIN_TRAIN_ROWS:
        raise ValueError(f"{asset}: insufficient pre-January training data")
    selection_origin = selection_training["ds"].max()
    selection_dates = observed_dates(full_data, SELECTION_START, SELECTION_END)
    if selection_dates.empty:
        raise ValueError(f"{asset}: no January 2022 observations for model selection")

    actual_selection = full_data[full_data["ds"].isin(selection_dates)][["ds", "y"]].rename(
        columns={"ds": "date", "y": "Actual"}
    )

    metrics: list[dict] = []
    errors: list[dict] = []

    for sentiment_columns, config_name in EXPERIMENTS:
        for model_name in MODEL_NAMES:
            print(
                f"  [SELECT] {asset} | {model_name} | {config_name}",
                flush=True,
            )
            try:
                forecast = run_model(
                    model_name,
                    full_data,
                    selection_training,
                    selection_origin,
                    selection_dates,
                    sentiment_columns,
                    compute_intervals=False,
                )
                evaluated = forecast.merge(actual_selection, on="date", how="left")
                score = calculate_metrics(evaluated["Actual"], evaluated["Predicted"])
                metrics.append(
                    {
                        "Asset": asset,
                        "Asset_Type": classify_asset(asset),
                        "Model": model_name,
                        "Config": config_name,
                        "Selection_Origin": selection_origin.strftime("%Y-%m-%d"),
                        **score,
                    }
                )

                safe_config = config_name.replace("+", "_").replace("-", "_").replace(" ", "_")
                evaluated.to_csv(
                    selection_forecast_directory
                    / f"{asset}_{model_name}_{safe_config}.csv",
                    index=False,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                metrics.append(
                    {
                        "Asset": asset,
                        "Asset_Type": classify_asset(asset),
                        "Model": model_name,
                        "Config": config_name,
                        "Selection_Origin": selection_origin.strftime("%Y-%m-%d"),
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "MAPE (%)": np.nan,
                        "NRMSE (%)": np.nan,
                        "R2": np.nan,
                        "Directional Accuracy (%)": np.nan,
                        "Error": str(exc),
                    }
                )
                errors.append(
                    {
                        "Asset": asset,
                        "Stage": "January selection",
                        "Model": model_name,
                        "Config": config_name,
                        "Error": str(exc),
                    }
                )

    metric_df = pd.DataFrame(metrics)
    valid = metric_df.dropna(subset=["RMSE"]).copy()
    if valid.empty:
        raise RuntimeError(f"{asset}: no valid January selection result")
    winner = valid.loc[valid["RMSE"].idxmin()].copy()

    winner_model = str(winner["Model"])
    winner_config = str(winner["Config"])
    winner_sentiment = next(
        columns for columns, name in EXPERIMENTS if name == winner_config
    )

    print(
        f"  [WINNER] {asset}: {winner_model} | {winner_config} | "
        f"January RMSE={winner['RMSE']:.6f}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Stage 2: Final blind February 2022 forecast
    # January is now known and may be included in final training.
    # No February Close, Volume or sentiment is passed to the model.
    # ------------------------------------------------------------
    final_available = full_data[full_data["ds"] < FINAL_FORECAST_START].copy()
    if len(final_available) < MIN_TRAIN_ROWS:
        raise ValueError(f"{asset}: insufficient data through January 2022")
    final_origin = final_available["ds"].max()

    # Final February dates are generated from the known DSE weekly calendar and
    # configured holidays. We intentionally do not inspect February market rows
    # from the uploaded files, even just to discover dates.
    future_dates = generate_dse_future_dates(
        FINAL_FORECAST_START, FINAL_FORECAST_END, DEFAULT_FINAL_HOLIDAYS
    )

    # This object contains no February market or sentiment observations.
    final_model_data = final_available.copy()
    final_forecast = run_model(
        winner_model,
        final_model_data,
        final_available,
        final_origin,
        future_dates,
        winner_sentiment,
    )

    # Defensive public-field protection: never write Actual to final website data.
    final_forecast = final_forecast[["date", "Predicted", "Lower", "Upper"]].copy()
    final_forecast.to_csv(final_forecast_directory / f"{asset}_forecast.csv", index=False)

    history = final_available[["ds", "y"]].rename(columns={"ds": "date", "y": "Close"})

    return {
        "asset": asset,
        "metrics": metrics,
        "winner": winner.to_dict(),
        "history": history,
        "final_forecast": final_forecast,
        "final_origin": final_origin,
        "errors": errors,
    }


def atomic_publish_app_data(
    paths: PipelinePaths,
    asset_results: list[dict],
) -> tuple[pd.DataFrame, dict]:
    app_dir = paths.app_data_directory
    parent = app_dir.parent
    temp_dir = parent / f"{app_dir.name}_new"
    backup_dir = parent / f"{app_dir.name}_previous"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    manifest_assets = []

    for result in sorted(asset_results, key=lambda x: x["asset"]):
        asset = result["asset"]
        winner = result["winner"]
        history = result["history"].copy()
        forecast = result["final_forecast"].copy()
        final_origin = pd.Timestamp(result["final_origin"])

        history.to_csv(temp_dir / f"{asset}_history.csv", index=False)
        forecast[["date", "Predicted", "Lower", "Upper"]].to_csv(
            temp_dir / f"{asset}_forecast.csv", index=False
        )

        forecast_start = pd.to_datetime(forecast["date"]).min().strftime("%Y-%m-%d")
        forecast_end = pd.to_datetime(forecast["date"]).max().strftime("%Y-%m-%d")
        selection_rmse = float(winner["RMSE"])
        selection_mape = float(winner["MAPE (%)"])

        summary_rows.append(
            {
                "Company": asset,
                "Asset_Type": classify_asset(asset),
                "Best_Model": winner["Model"],
                "Best_Experiment": winner["Config"],
                # These errors are January model-selection values, not February errors.
                "Selection_RMSE": selection_rmse,
                "Selection_MAPE": selection_mape,
                # Backward-compatible names used by older versions of the website.
                "RMSE": selection_rmse,
                "MAPE": selection_mape,
                "Selection_Period": "2022-01-01 to 2022-01-31",
                "Last_History_Date": final_origin.strftime("%Y-%m-%d"),
                "Forecast_Start": forecast_start,
                "Forecast_End": forecast_end,
            }
        )
        manifest_assets.append(
            {
                "company": asset,
                "asset_type": classify_asset(asset),
                "selected_model": winner["Model"],
                "selected_experiment": winner["Config"],
                "selection_rmse": selection_rmse,
                "cutoff": final_origin.strftime("%Y-%m-%d"),
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "rows": int(len(forecast)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(temp_dir / "companies_summary.csv", index=False)

    now = datetime.now(timezone.utc)
    manifest = {
        "run_id": now.strftime("forecast-%Y%m%d-%H%M%S"),
        "status": "completed",
        "generated_at_utc": now.isoformat(),
        "selection_period": {
            "start": SELECTION_START.strftime("%Y-%m-%d"),
            "end": SELECTION_END.strftime("%Y-%m-%d"),
            "purpose": "model and sentiment configuration selection",
        },
        "forecast_period": {
            "start": FINAL_FORECAST_START.strftime("%Y-%m-%d"),
            "end": FINAL_FORECAST_END.strftime("%Y-%m-%d"),
            "purpose": "final blind one-month forecast",
        },
        "future_information_rule": (
            "February Close, Volume, and sentiment are not used in final model fitting or forecasting."
        ),
        "asset_count": len(manifest_assets),
        "assets": manifest_assets,
    }
    with (temp_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    # Atomic-ish package switch. The old package remains until the new package is complete.
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    try:
        if app_dir.exists():
            app_dir.rename(backup_dir)
        temp_dir.rename(app_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if app_dir.exists() and app_dir != temp_dir:
            shutil.rmtree(app_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.rename(app_dir)
        raise

    return summary, manifest


def run_pipeline(paths: PipelinePaths | None = None, progress_callback=None):
    paths = paths or resolve_paths()
    validate_input_paths(paths)

    paths.output_directory.mkdir(parents=True, exist_ok=True)
    selection_forecast_directory = paths.output_directory / "selection_forecasts"
    final_forecast_directory = paths.output_directory / "final_forecasts"
    selection_forecast_directory.mkdir(parents=True, exist_ok=True)
    final_forecast_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("DSE SENTIMENT-AWARE FORECAST PIPELINE", flush=True)
    print("January 2022 = model/configuration selection", flush=True)
    print("February 2022 = final blind website forecast", flush=True)
    print("=" * 78, flush=True)
    print(f"Price directory: {paths.price_directory}", flush=True)
    print(f"BanglaBERT file: {paths.banglabert_file}", flush=True)
    print(f"Lexicon file: {paths.lexicon_file}", flush=True)
    print(f"Website data: {paths.app_data_directory}", flush=True)

    bangla_feat, lexicon_feat = load_sentiment(paths)
    companies = find_asset_files(paths.price_directory)

    all_asset_results: list[dict] = []
    top_errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                process_asset,
                asset,
                filepath,
                bangla_feat,
                lexicon_feat,
                selection_forecast_directory,
                final_forecast_directory,
            ): asset
            for asset, filepath in companies.items()
        }

        completed = 0
        for future in as_completed(future_map):
            asset = future_map[future]
            completed += 1
            try:
                result = future.result()
                all_asset_results.append(result)
                print(f"[{completed}/{len(future_map)}] {asset} COMPLETE", flush=True)
                if progress_callback is not None:
                    progress_callback(completed, len(future_map), asset, "complete")
            except Exception as exc:
                print(f"[{completed}/{len(future_map)}] {asset} FAILED: {exc}", flush=True)
                if progress_callback is not None:
                    progress_callback(completed, len(future_map), asset, "failed")
                top_errors.append({"Asset": asset, "Stage": "asset pipeline", "Error": str(exc)})

    if not all_asset_results:
        raise RuntimeError("No asset completed successfully")

    all_metrics = []
    all_errors = list(top_errors)
    winners = []
    for result in all_asset_results:
        all_metrics.extend(result["metrics"])
        all_errors.extend(result["errors"])
        winners.append(result["winner"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(paths.output_directory / "january_selection_results.csv", index=False)

    winners_df = pd.DataFrame(winners).sort_values("Asset").reset_index(drop=True)
    winners_df.to_csv(paths.output_directory / "best_january_selection_each_asset.csv", index=False)

    if all_errors:
        pd.DataFrame(all_errors).to_csv(paths.output_directory / "errors.csv", index=False)

    summary, manifest = atomic_publish_app_data(paths, all_asset_results)

    print("\nFINAL SELECTED MODELS / CONFIGURATIONS", flush=True)
    print(
        summary[
            ["Company", "Best_Model", "Best_Experiment", "RMSE", "Forecast_Start", "Forecast_End"]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nPublished website package: {paths.app_data_directory}", flush=True)
    return summary, manifest


if __name__ == "__main__":
    run_pipeline()
