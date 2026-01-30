import pandas as pd
import numpy as np

def ensure_schema(df, stage="signals"):
    """
    確保 DataFrame 包含所有必要欄位
    
    Args:
        df: 輸入的 DataFrame
        stage: 處理階段 (用於未來擴展)
    
    Returns:
        DataFrame: 補齊欄位後的資料
    """
    df = df.copy()
    
    # 定義所有可能用到的欄位,預設為 NaN 或 False
    base_cols = {
        # OHLCV 基礎欄位
        "Close": np.nan,
        "Open": np.nan,
        "High": np.nan,
        "Low": np.nan,
        "Volume": np.nan,
        
        # 均線
        "MA5": np.nan,
        "MA10": np.nan,
        "MA20": np.nan,
        "MA60": np.nan,
        "MA240": np.nan,
        
        # 量能
        "VOL_MA": np.nan,
        
        # 技術指標
        "K": np.nan,
        "D": np.nan,
        "RSI": np.nan,
        "DIF": np.nan,
        "DEA": np.nan,
        "MACD_Hist": np.nan,
        
        # 評分與狀態
        "Tech_Score": 0,
        "Position": 0,
        
        # 訊號
        "Buy_Signal": False,
        "Sell_Signal": False,
        "Sell_Core": False,
        "Sell_StopLine": False,
        
        # 原因標記
        "Buy_Reason": "NONE",
        "Sell_Reason_Raw": "NONE"
    }

    # 補齊缺少的欄位
    for col, default_val in base_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # 確保布林值欄位正確型別
    bool_cols = ["Buy_Signal", "Sell_Signal", "Sell_Core", "Sell_StopLine"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


def ensure_ohlcv(df):
    """
    確保 DataFrame 具備標準 OHLCV 格式
    
    處理 yfinance 下載的 MultiIndex 格式,並統一欄位命名
    
    Args:
        df: 原始下載的 DataFrame
    
    Returns:
        DataFrame: 標準化的 OHLCV 資料
    
    Raises:
        ValueError: 缺少必要欄位時
    """
    df = df.copy()
    
    # 處理 MultiIndex (yfinance 格式)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]

    # 建立欄位對應字典 (不區分大小寫)
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]

    # 檢查並重命名欄位
    for n in need:
        if n not in cols:
            if n == "volume":
                # Volume 欄位可能缺失,補上 NaN
                df["Volume"] = np.nan
                cols["volume"] = "Volume"
            else:
                raise ValueError(f"缺少必要欄位: {n.upper()}, 無法進行分析")

    # 提取並重命名為標準格式
    out = df[[cols[n] for n in need]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # 清理資料
    out = out.dropna(subset=["Close"])  # 移除收盤價為空的列
    out.index = pd.to_datetime(out.index)  # 確保索引為日期格式
    
    return out


def safe_fillna(series, default=0):
    """
    安全填補 NaN 值
    
    Args:
        series: 要處理的 Series
        default: 預設填補值
    
    Returns:
        Series: 填補後的資料
    """
    return series.fillna(default)


def calculate_drawdown(equity_curve):
    """
    計算回撤曲線
    
    Args:
        equity_curve: 權益曲線 (Series 或 array)
    
    Returns:
        Series: 回撤百分比
    """
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)
    
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1)
    
    return drawdown


def calculate_returns(prices):
    """
    計算報酬率序列
    
    Args:
        prices: 價格序列 (Series 或 array)
    
    Returns:
        Series: 報酬率序列
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    return prices.pct_change().fillna(0)
