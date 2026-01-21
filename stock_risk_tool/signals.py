import numpy as np
import pandas as pd

def generate_signals(df, p, mode, stock_type="DEFAULT"):
    df = df.copy()

    # 1. 定義生命線 (StopLine)
    # 這是最後防線，跌破這裡就是「趨勢翻空」
    if stock_type == "MOMENTUM": stop_col = "MA10"
    elif stock_type == "WEIGHT": stop_col = "MA20"
    elif stock_type == "FINANCE": stop_col = "MA60"
    else: stop_col = "MA20"

    if stop_col not in df.columns: stop_col = "MA20"
    df["StopLine"] = df[stop_col]

    # 計算乖離率 (Bias)
    # 用來判斷是否「漲太多」
    df["Bias"] = (df["Close"] - df["MA20"]) / df["MA20"]

    # ========================================================
    # 2. 買進訊號 (維持不變)
    # ========================================================
    ma_lookback = 5
    slope = (df[stop_col] - df[stop_col].shift(ma_lookback)) / df[stop_col].shift(ma_lookback)
    slope_threshold = p.get("MA_SLOPE_THRESHOLD", 0.005)
    is_trend_up = slope > slope_threshold
    df["MA_Slope"] = slope

    trend_strong = df["SanYang"] | df["SiHai"]
    vol_ok = df["Volume"] > df["VOL_MA"]
    if stock_type == "FINANCE" and "MA240" in df.columns:
        long_term_ok = df["Close"] > df["MA240"]
    else:
        long_term_ok = True

    df["Buy_Signal"] = (df["Close"] > df["StopLine"]) & is_trend_up & trend_strong & long_term_ok
    df["Buy_Reason"] = np.where(df["SiHai"], "FOUR_SEAS", np.where(df["SanYang"], "THREE_SUNS", "TREND"))

    # ========================================================
    # 3. 賣出訊號 (Sell Logic) - 三層次賣出法
    # ========================================================

    # --- 層次 1: 高檔獲利了結 (Profit Taking) ---
    # 邏輯：乖離過大 (>15%) 且 (跌破5日線 或 KD高檔死叉)
    # 這是為了「保住獲利」，不要抱上山又抱下山

    is_high_bias = df["Bias"] > 0.15  # 乖離率 > 15% (可調整)
    break_ma5 = df["Close"] < df["MA5"]
    kd_dead_cross = (df["K"] > 80) & (df["K"] < df["D"]) # 高檔死叉

    sell_profit = is_high_bias & (break_ma5 | kd_dead_cross)

    # --- 層次 2: 趨勢轉弱 (Technical Exit) ---
    # 邏輯：跌破生命線 (MA10/MA20)
    # 這是波段結束的訊號

    buffer = p.get("STOP_BUFFER_PCT", 0.015)
    sell_stop_line = df["Close"] < (df["StopLine"] * (1 - buffer))

    # --- 層次 3: 結構破壞 (Structural Break) ---
    # 邏輯：跌破爆量低點 或 假突破
    # 這是「兇多吉少」的逃命訊號

    sell_big_vol = (df["Close"] < df["BigVol_Low"]) & df["BigVol_Low"].notna()

    is_black = df["Close"] < df["Open"]
    fake_break = (df["High"].shift(1) > df["High"].shift(2)) & is_black & (df["Close"] < df["MA5"])

    # --- 保護機制 (Protections) ---
    # 防止在 V 型反轉底部賣出 (只針對 StopLine 有效，BigVol 不保護)

    is_below_ma = df["Close"] < df["StopLine"]
    days_below = is_below_ma.rolling(3).sum()
    tech_breakdown = (days_below >= 3) # 跌破3天才算真跌破

    rsi_protect = df["RSI"] < 45 # 超賣保護
    ma60_trend_up = df["MA60"] > df["MA60"].shift(20)
    ma60_support = df["Close"] > (df["MA60"] * 0.96)
    trend_protect = ma60_trend_up & ma60_support # 季線保護

    # 硬停損 (Hard Stop)：跌破季線 5% 一定要跑，不管有沒有保護
    hard_stop = df["Close"] < (df["MA60"] * 0.95)

    # 綜合賣訊邏輯：
    # 1. 高檔獲利 (Profit) -> 優先執行
    # 2. 結構破壞 (BigVol/Fake) -> 優先執行
    # 3. 趨勢轉弱 (StopLine) -> 需經過「3日原則」且「無保護」才執行
    # 4. 硬停損 (Hard Stop) -> 無條件執行

    real_sell_stopline = tech_breakdown & ~rsi_protect & ~trend_protect

    df["Sell_Signal"] = sell_profit | \
                        sell_big_vol | \
                        fake_break | \
                        real_sell_stopline | \
                        hard_stop

    # 標記賣出原因 (優先級：硬停損 > 爆量破 > 獲利了結 > 破線)
    conds = [hard_stop, sell_big_vol, sell_profit, fake_break, real_sell_stopline]
    choices = ["HARD_STOP", "BIG_VOL", "TAKE_PROFIT", "FAKE", "MA_BREAK"]
    df["Sell_Reason_Raw"] = np.select(conds, choices, default="NONE")

    # 狀態標記
    df["In_Protection"] = tech_breakdown & (rsi_protect | trend_protect)

    # 評分 (加入獲利了結扣分)
    score = pd.Series(50, index=df.index)
    score += np.where(df["SanYang"], 20, 0)
    score += np.where(is_trend_up, 10, 0)
    score += np.where(df["Close"] < df["StopLine"], -20, 0)
    score += np.where(sell_big_vol, -30, 0)
    score += np.where(sell_profit, -10, 0) # 獲利了結也算轉弱訊號
    df["Tech_Score"] = score.clip(0, 100)

    df["Sell_Core"] = sell_big_vol
    df["Sell_StopLine"] = df["Sell_Signal"]
    df["Near_StopLine_Warn"] = (days_below == 2)
    df["Short_Signal"] = False

    return df
