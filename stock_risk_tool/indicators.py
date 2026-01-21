import pandas as pd
import numpy as np

def compute_atr_pct(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / close

def add_indicators(df, p):
    df = df.copy()

    # 均線
    for ma in ["MA5", "MA10", "MA20", "MA60", "MA240"]:
        if p.get(ma): df[ma] = df["Close"].rolling(p[ma]).mean()

    df["VOL_MA"] = df["Volume"].rolling(p["VOL_MA"]).mean()
    df["ATRp"] = compute_atr_pct(df, p["ATR_N"])

    # 輔助指標 (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(p["RSI_N"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(p["RSI_N"]).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs)).fillna(50)

    # KD
    low_min = df["Low"].rolling(p["KD_N"]).min()
    high_max = df["High"].rolling(p["KD_N"]).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()

    # MACD
    ema_fast = df["Close"].ewm(span=p["MACD_FAST"]).mean()
    ema_slow = df["Close"].ewm(span=p["MACD_SLOW"]).mean()
    df["MACD_Hist"] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=p["MACD_SIGNAL"]).mean()

    # OBV
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # --- 老王戰法核心 ---
    # 1. 爆大量與低點
    is_big_vol = (df["Volume"] > df["VOL_MA"] * p["BIGVOL_MULT"])
    df["Is_Big_Vol"] = is_big_vol
    df["BigVol_Low"] = np.where(is_big_vol, df["Low"], np.nan)
    df["BigVol_Low"] = df["BigVol_Low"].ffill() # 向後延伸支撐線

    # 2. 長紅與缺口
    body_pct = (df["Close"] - df["Open"]) / df["Open"]
    df["Is_Big_Red"] = (body_pct > p["BIG_RED_BODY_PCT"])
    df["Gap_Up"] = df["Low"] > df["High"].shift(1)

    # 3. 三陽開泰 / 四海遊龍
    df["SanYang"] = (df["Close"] > df["MA5"]) & (df["Close"] > df["MA10"]) & (df["Close"] > df["MA20"])
    df["SiHai"] = df["SanYang"] & (df["Close"] > df["MA60"])

    # 兼容性填充
    df["GapUp"] = df["Gap_Up"]
    df["Gap_Support"] = False
    df["Breakout"] = False
    df["Wash"] = False
    df["BigVol_Confirmed"] = True

    return df

def classify_mode(df, p):
    # 為了讓 Main 不報錯，回傳完整結構
    return {"mode": "老王戰法", "atr_med": 0, "long_ratio": 0, "yokai": False}
