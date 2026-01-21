import pandas as pd
import numpy as np

def ensure_schema(df, stage="signals"):
    df = df.copy()
    # 定義所有可能用到的欄位，預設為 NaN 或 False
    base_cols = {
        "Close": np.nan, "MA5": np.nan, "MA20": np.nan, "MA60": np.nan,
        "Volume": np.nan, "VOL_MA": np.nan,
        "K": np.nan, "D": np.nan, "RSI": np.nan,
        "DIF": np.nan, "DEA": np.nan, "MACD_Hist": np.nan,
        "Tech_Score": 0, # [新增] 技術評分
        "Position": 0, "Buy_Signal": False, "Sell_Signal": False,
        "Sell_Core": False, "Sell_StopLine": False,
        "Buy_Reason": "NONE", "Sell_Reason_Raw": "NONE"
    }

    for col, default_val in base_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # 確保布林值欄位正確
    bool_cols = ["Buy_Signal", "Sell_Signal", "Sell_Core", "Sell_StopLine"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df

def ensure_ohlcv(df):
    df = df.copy()
    # 處理 MultiIndex (如果是 yfinance 下載的格式)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]

    cols = {c.lower(): c for c in df.columns}
    need = ["open","high","low","close","volume"]

    # 檢查並重命名欄位
    for n in need:
        if n not in cols:
            if n == "volume":
                df["Volume"] = np.nan
                cols["volume"] = "Volume"
            else:
                raise ValueError(f"缺少欄位 {n}，無法進行分析")

    out = df[[cols[n] for n in need]].copy()
    out.columns = ["Open","High","Low","Close","Volume"]
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    return out
