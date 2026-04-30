"""Price history fetcher and ML feature engineering via yfinance.

Provides two public functions consumed by QuantitativeAgent:
    fetch_price_history(ticker)  → pd.DataFrame | None
    run_ml_prediction(df)        → dict
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum trading days required for meaningful feature + target engineering
MIN_ROWS = 120

# Features fed into the RandomForest
FEATURE_COLS: list[str] = [
    "price_to_ma20",
    "price_to_ma50",
    "ma20_above_ma50",
    "ma50_above_ma200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "atr_14",
    "vol_z",
]


# ─── Data fetching ─────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, years: int = 3) -> pd.DataFrame | None:
    """Download adjusted daily OHLCV data from Yahoo Finance via yfinance.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        years:  How many years of history to request (default 3).

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by date,
        or None if the download fails or returns insufficient data.
    """
    import yfinance as yf  # lazy import — optional dependency

    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)  # small buffer for weekends/holidays

    try:
        df: pd.DataFrame = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if df is None or df.empty:
        logger.warning("yfinance returned empty DataFrame for %s", ticker)
        return None

    # Flatten multi-level columns (yfinance sometimes returns these for single tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only standard OHLCV columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        logger.warning("yfinance columns incomplete for %s: %s", ticker, list(df.columns))
        return None

    df = df[list(required)].dropna(subset=["Close"])

    if len(df) < MIN_ROWS:
        logger.warning(
            "Insufficient price history for %s: %d rows (need %d)",
            ticker, len(df), MIN_ROWS,
        )
        return None

    logger.info(
        "Price history fetched for %s: %d rows from %s to %s",
        ticker, len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


# ─── Feature engineering ───────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and the 20-day forward return target.

    All features are stored as new columns on a copy of df.

    Target labels (column 'target'):
        'UP'   — forward 20-day return >  +3 %
        'FLAT' — forward 20-day return ± 3 %
        'DOWN' — forward 20-day return < -3 %
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    # ── Moving averages & distance from price ──
    df["ma_20"]  = close.rolling(20).mean()
    df["ma_50"]  = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()

    df["price_to_ma20"]    = close / df["ma_20"] - 1
    df["price_to_ma50"]    = close / df["ma_50"] - 1
    df["ma20_above_ma50"]  = (df["ma_20"]  > df["ma_50"]).astype(float)
    df["ma50_above_ma200"] = (df["ma_50"]  > df["ma_200"]).astype(float)

    # ── RSI (14) ──
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ── MACD (12/26/9) ──
    ema12         = close.ewm(span=12, adjust=False).mean()
    ema26         = close.ewm(span=26, adjust=False).mean()
    df["macd"]         = ema12 - ema26
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20, 2σ) ──
    bb_mid     = close.rolling(20).mean()
    bb_std     = close.rolling(20).std()
    bb_upper   = bb_mid + 2 * bb_std
    bb_lower   = bb_mid - 2 * bb_std
    bb_range   = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_range  # 0 = lower band, 1 = upper band

    # ── Momentum (price returns over N days) ──
    df["mom_5d"]  = close.pct_change(5)
    df["mom_20d"] = close.pct_change(20)
    df["mom_60d"] = close.pct_change(60)

    # ── Volatility: normalised Average True Range ──
    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean() / close  # normalised

    # ── Volume z-score (unusual volume signal) ──
    vol_mean     = vol.rolling(20).mean()
    vol_std      = vol.rolling(20).std().replace(0, np.nan)
    df["vol_z"]  = (vol - vol_mean) / vol_std

    # ── Target: 20-day forward return → direction label ──
    df["forward_return_20d"] = close.shift(-20) / close - 1
    df["target"] = pd.cut(
        df["forward_return_20d"],
        bins=[-np.inf, -0.03, 0.03, np.inf],
        labels=["DOWN", "FLAT", "UP"],
    )

    return df


# ─── ML prediction pipeline ────────────────────────────────────────────────────

def run_ml_prediction(df: pd.DataFrame) -> dict[str, Any]:
    """Train a RandomForestClassifier on historical features, predict current direction.

    Pipeline:
        1. Engineer features + target on the full DataFrame.
        2. Drop rows with NaN features or target.
        3. Time-ordered 80/20 train/test split (no shuffling to avoid look-ahead).
        4. Fit RandomForest on train set.
        5. Evaluate holdout accuracy on test set.
        6. Predict direction for the latest (most recent) row.
        7. Return prediction, probabilities, top feature importances, and a
           technical snapshot of current indicators.

    Args:
        df: Raw OHLCV DataFrame from fetch_price_history().

    Returns:
        Dict with keys:
            prediction          — "UP" | "FLAT" | "DOWN"
            probabilities       — {"UP": float, "FLAT": float, "DOWN": float}
            holdout_accuracy    — float [0, 1]
            holdout_samples     — int
            training_samples    — int
            top_features        — list of (feature_name, importance) tuples
            current_snapshot    — dict of latest indicator values
            data_start / data_end — ISO date strings
        Or on failure:
            error               — short error code string
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        logger.error("scikit-learn not installed: %s", exc)
        return {"error": "scikit_learn_missing"}

    # ── Feature engineering ──
    df_feat = _engineer_features(df)
    df_clean = df_feat[FEATURE_COLS + ["target"]].dropna()

    if len(df_clean) < 60:
        logger.warning("Only %d clean rows after feature engineering", len(df_clean))
        return {"error": "insufficient_clean_rows", "rows_available": len(df_clean)}

    X = df_clean[FEATURE_COLS].values.astype(float)
    y = df_clean["target"].astype(str).values

    # ── Time-ordered train/test split ──
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(set(y_train)) < 2:
        logger.warning("Training set has only one class — cannot fit classifier")
        return {"error": "single_class_train"}

    # ── Fit model ──
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",  # handle class imbalance gracefully
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Holdout evaluation ──
    y_pred   = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # ── Predict on latest row ──
    latest_features = df_feat[FEATURE_COLS].dropna().iloc[-1].values.reshape(1, -1)
    predicted_class = str(clf.predict(latest_features)[0])
    proba_raw       = clf.predict_proba(latest_features)[0]
    classes         = list(clf.classes_)
    probabilities   = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba_raw)}

    # ── Feature importances ──
    importances  = sorted(
        zip(FEATURE_COLS, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    top_features = [(name, round(float(imp), 4)) for name, imp in importances[:5]]

    # ── Current technical snapshot ──
    latest = df_feat.iloc[-1]

    def _safe(key: str, decimals: int = 4) -> float | None:
        val = latest.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)

    current_snapshot = {
        "close_price":   round(float(df["Close"].iloc[-1]), 2),
        "rsi_14":        _safe("rsi_14", 2),
        "macd_hist":     _safe("macd_hist", 4),
        "bb_position":   _safe("bb_position", 4),
        "price_to_ma20": _safe("price_to_ma20", 4),
        "price_to_ma50": _safe("price_to_ma50", 4),
        "mom_20d":       _safe("mom_20d", 4),
        "mom_60d":       _safe("mom_60d", 4),
        "vol_z":         _safe("vol_z", 4),
    }

    return {
        "prediction":       predicted_class,
        "probabilities":    probabilities,
        "holdout_accuracy": round(accuracy, 4),
        "holdout_samples":  int(len(y_test)),
        "training_samples": int(split),
        "top_features":     top_features,
        "current_snapshot": current_snapshot,
        "data_start":       str(df.index[0].date()),
        "data_end":         str(df.index[-1].date()),
    }
