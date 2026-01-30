import pandas as pd
import numpy as np

def compute_atr_pct(df, n=14):
    """
    計算 ATR 百分比 (相對於收盤價)
    
    Args:
        df: 包含 OHLC 的 DataFrame
        n: ATR 週期
    
    Returns:
        Series: ATR 百分比
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    
    # 計算真實波幅 (True Range)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # 回傳 ATR 百分比
    return tr.rolling(n).mean() / close


def add_indicators(df, p):
    """
    新增所有技術指標
    
    Args:
        df: 原始 OHLC DataFrame
        p: 參數字典
    
    Returns:
        DataFrame: 加入指標後的資料
    """
    df = df.copy()

    # ========================================================
    # 1. 均線系統
    # ========================================================
    for ma in ["MA5", "MA10", "MA20", "MA60", "MA240"]:
        if p.get(ma):
            df[ma] = df["Close"].rolling(p[ma]).mean()

    # ========================================================
    # 2. 量能指標
    # ========================================================
    df["VOL_MA"] = df["Volume"].rolling(p["VOL_MA"]).mean()
    
    # ========================================================
    # 3. 波動率指標
    # ========================================================
    df["ATRp"] = compute_atr_pct(df, p["ATR_N"])

    # ========================================================
    # 4. RSI (相對強弱指標)
    # ========================================================
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(p["RSI_N"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(p["RSI_N"]).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs)).fillna(50)

    # ========================================================
    # 5. KD 指標
    # ========================================================
    low_min = df["Low"].rolling(p["KD_N"]).min()
    high_max = df["High"].rolling(p["KD_N"]).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()

    # ========================================================
    # 6. MACD
    # ========================================================
    ema_fast = df["Close"].ewm(span=p["MACD_FAST"]).mean()
    ema_slow = df["Close"].ewm(span=p["MACD_SLOW"]).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=p["MACD_SIGNAL"]).mean()
    df["MACD_Hist"] = macd_line - signal_line

    # ========================================================
    # 7. OBV (能量潮指標)
    # ========================================================
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # ========================================================
    # 8. 老王戰法核心指標
    # ========================================================
    
    # 8.1 爆大量與低點
    is_big_vol = (df["Volume"] > df["VOL_MA"] * p["BIGVOL_MULT"])
    df["Is_Big_Vol"] = is_big_vol
    
    # 爆量低點支撐線 (向後延伸)
    df["BigVol_Low"] = np.where(is_big_vol, df["Low"], np.nan)
    df["BigVol_Low"] = df["BigVol_Low"].ffill()

    # 8.2 長紅與缺口
    body_pct = (df["Close"] - df["Open"]) / df["Open"]
    df["Is_Big_Red"] = (body_pct > p["BIG_RED_BODY_PCT"])
    df["Gap_Up"] = df["Low"] > df["High"].shift(1)

    # 8.3 三陽開泰 / 四海遊龍
    df["SanYang"] = (
        (df["Close"] > df["MA5"]) & 
        (df["Close"] > df["MA10"]) & 
        (df["Close"] > df["MA20"])
    )
    df["SiHai"] = df["SanYang"] & (df["Close"] > df["MA60"])

    # ========================================================
    # 9. 相容性欄位 (保持與舊版一致)
    # ========================================================
    df["GapUp"] = df["Gap_Up"]
    df["Gap_Support"] = False
    df["Breakout"] = False
    df["Wash"] = False
    df["BigVol_Confirmed"] = True

    return df


def classify_mode(df, p):
    """
    市場模式分類 (簡化版)
    
    Args:
        df: 包含技術指標的 DataFrame
        p: 參數字典
    
    Returns:
        dict: 市場模式資訊
    """
    # 為了讓 Main 不報錯,回傳完整結構
    # 未來可擴展為真正的市場分類邏輯
    
    return {
        "mode": "老王戰法",
        "atr_med": 0,
        "long_ratio": 0,
        "yokai": False
    }
