import numpy as np
import pandas as pd

def _future_window_max(series, lookahead):
    reversed_series = series.iloc[::-1]
    future_max = (
        reversed_series.rolling(window=lookahead, min_periods=1)
        .max()
        .iloc[::-1]
    )
    return future_max


def detect_sell_before_rise(
    df,
    lookahead=3,
    rise_threshold=0.02,
    price_col="Close",
    signal_col="Sell_Signal",
    use_high=False,
    high_col="High",
):
    """
    偵測賣出後短期內價格上漲 (賣在起漲前) 的情況。

    Args:
        df: 包含價格與賣出訊號的 DataFrame
        lookahead: 觀察未來幾根 (天/週等) K 線
        rise_threshold: 未來最大漲幅閾值 (如 0.02 = 2%)
        price_col: 當前價格欄位 (用於計算漲幅)
        signal_col: 賣出訊號欄位名稱
        use_high: 是否使用 High 來判斷未來最大漲幅
        high_col: 最高價欄位名稱

    Returns:
        tuple: (sell_before_rise_flag, rise_pct)
    """
    future_col = high_col if use_high else price_col
    future_max = _future_window_max(df[future_col].shift(-1), lookahead)
    rise_pct = (future_max / df[price_col]) - 1
    sell_before_rise = df[signal_col] & (rise_pct >= rise_threshold)
    return sell_before_rise.fillna(False), rise_pct


def generate_signals(df, p, mode, stock_type="DEFAULT"):
    """
    產生買賣訊號 - 重構版
    
    核心改進:
    1. 簡化賣出邏輯為「兩層式」結構
    2. 明確的保護機制觸發條件
    3. 可調整的參數化設計
    
    Args:
        df: 包含技術指標的 DataFrame
        p: 參數字典
        mode: 市場模式
        stock_type: 股票類型
    
    Returns:
        DataFrame: 加入買賣訊號後的資料
    """
    df = df.copy()

    # ========================================================
    # 1. 定義生命線 (StopLine) - 依股票類型調整
    # ========================================================
    if stock_type == "MOMENTUM":
        stop_col = "MA10"      # 動能股:快速反應
    elif stock_type == "WEIGHT":
        stop_col = "MA20"      # 權值股:月線支撐
    elif stock_type == "FINANCE":
        stop_col = "MA60"      # 金融股:季線為主
    else:
        stop_col = "MA20"      # 預設

    # 確保欄位存在
    if stop_col not in df.columns:
        stop_col = "MA20"
    
    df["StopLine"] = df[stop_col]

    # 計算乖離率 (用於判斷過熱)
    df["Bias"] = (df["Close"] - df["MA20"]) / df["MA20"]

    # ========================================================
    # 2. 買進訊號 - 維持原邏輯
    # ========================================================
    
    # 均線斜率判斷 (趨勢向上)
    ma_lookback = 5
    slope = (df[stop_col] - df[stop_col].shift(ma_lookback)) / df[stop_col].shift(ma_lookback)
    slope_threshold = p.get("MA_SLOPE_THRESHOLD", 0.005)
    is_trend_up = slope > slope_threshold
    df["MA_Slope"] = slope

    # 型態強勢
    trend_strong = df["SanYang"] | df["SiHai"]
    
    # 量能確認
    vol_ok = df["Volume"] > df["VOL_MA"]
    
    # 長期趨勢 (金融股特別要求)
    if stock_type == "FINANCE" and "MA240" in df.columns:
        long_term_ok = df["Close"] > df["MA240"]
    else:
        long_term_ok = True

    # 綜合買進條件
    df["Buy_Signal"] = (
        (df["Close"] > df["StopLine"]) & 
        is_trend_up & 
        trend_strong & 
        long_term_ok
    )
    
    # 買進原因標記
    df["Buy_Reason"] = np.where(
        df["SiHai"], "FOUR_SEAS", 
        np.where(df["SanYang"], "THREE_SUNS", "TREND")
    )

    # ========================================================
    # 3. 賣出訊號 - 重構為「兩層式」
    # ========================================================
    
    # --- 參數設定 (可調整) ---
    bias_threshold = p.get("BIAS_THRESHOLD", 0.15)      # 乖離閾值 15%
    stop_buffer = p.get("STOP_BUFFER_PCT", 0.015)       # 停損緩衝 1.5%
    hard_stop_pct = p.get("HARD_STOP_PCT", 0.05)        # 硬停損 5%
    
    # ========================================================
    # 第一層:主要賣出訊號 (輕度/中度賣壓)
    # ========================================================
    
    # A. 獲利了結 (高檔賣壓)
    # 條件:乖離過大 + (跌破5日線 或 KD高檔死叉)
    is_high_bias = df["Bias"] > bias_threshold
    break_ma5 = df["Close"] < df["MA5"]
    kd_dead_cross = (df["K"] > 80) & (df["K"] < df["D"])
    
    sell_profit = is_high_bias & (break_ma5 | kd_dead_cross)
    
    # B. 結構破壞 (爆量低點跌破)
    # 條件:收盤 < 爆量低點
    sell_big_vol = (df["Close"] < df["BigVol_Low"]) & df["BigVol_Low"].notna()
    
    # C. 假突破 (誘多)
    # 條件:前日創高 + 今日黑K + 收盤跌破5日線
    is_black = df["Close"] < df["Open"]
    fake_break = (
        (df["High"].shift(1) > df["High"].shift(2)) & 
        is_black & 
        (df["Close"] < df["MA5"])
    )
    
    # ========================================================
    # 第二層:趨勢停損 (需經過保護機制)
    # ========================================================
    
    # D. 生命線停損 (需確認)
    # 條件:跌破生命線 + 持續 N 天
    is_below_stopline = df["Close"] < (df["StopLine"] * (1 - stop_buffer))
    
    # 計算連續跌破天數 (3日原則)
    days_below = is_below_stopline.rolling(3).sum()
    tech_breakdown = (days_below >= 3)
    
    # --- 保護機制 (避免 V 型反轉誤殺) ---
    
    # 保護1: RSI 超賣保護 (避免殺在底部)
    rsi_protect = df["RSI"] < 40  # 從 45 下調到 40 (更嚴格)
    
    # 保護2: 季線趨勢保護 (大趨勢仍向上)
    ma60_trend_up = df["MA60"] > df["MA60"].shift(20)
    ma60_support = df["Close"] > (df["MA60"] * 0.97)  # 3% 容忍度
    trend_protect = ma60_trend_up & ma60_support
    
    # 綜合保護 (任一條件成立就保護)
    is_protected = rsi_protect | trend_protect
    
    # 最終趨勢停損 = 技術破位 + 沒有保護
    sell_stopline = tech_breakdown & ~is_protected
    
    # ========================================================
    # 硬停損 (無條件執行)
    # ========================================================
    
    # E. 硬停損 (季線下破 5%)
    # 這是最後防線,不管任何保護都要跑
    hard_stop = df["Close"] < (df["MA60"] * (1 - hard_stop_pct))
    
    # ========================================================
    # 綜合賣出訊號
    # ========================================================
    
    df["Sell_Signal"] = (
        sell_profit |      # 獲利了結
        sell_big_vol |     # 結構破壞
        fake_break |       # 假突破
        sell_stopline |    # 趨勢停損
        hard_stop          # 硬停損
    )

    # ========================================================
    # 賣在起漲前判斷
    # ========================================================
    sell_lookahead = p.get("SELL_LOOKAHEAD", 3)
    sell_rise_threshold = p.get("SELL_PREMATURE_THRESHOLD", 0.02)
    sell_use_high = p.get("SELL_PREMATURE_USE_HIGH", False)
    sell_before_rise, sell_rise_pct = detect_sell_before_rise(
        df,
        lookahead=sell_lookahead,
        rise_threshold=sell_rise_threshold,
        use_high=sell_use_high,
    )
    df["Sell_Premature"] = sell_before_rise
    df["Sell_Premature_RisePct"] = sell_rise_pct.fillna(0)
    
    # ========================================================
    # 賣出原因標記 (優先級排序)
    # ========================================================
    conditions = [
        hard_stop,
        sell_big_vol,
        sell_profit,
        fake_break,
        sell_stopline
    ]
    
    choices = [
        "HARD_STOP",      # 硬停損 (最嚴重)
        "BIG_VOL_BREAK",  # 爆量破位
        "TAKE_PROFIT",    # 獲利了結
        "FAKE_BREAK",     # 假突破
        "MA_BREAK"        # 均線破位
    ]
    
    df["Sell_Reason_Raw"] = np.select(conditions, choices, default="NONE")
    
    # ========================================================
    # 輔助標記 (用於診斷)
    # ========================================================
    
    # 保護中狀態 (有破位但受保護)
    df["In_Protection"] = tech_breakdown & is_protected
    
    # 接近停損警告 (跌破 2 天)
    df["Near_StopLine_Warn"] = (days_below == 2)
    
    # 個別賣出訊號 (用於分析)
    df["Sell_Profit"] = sell_profit
    df["Sell_BigVol"] = sell_big_vol
    df["Sell_Fake"] = fake_break
    df["Sell_StopLine_Raw"] = tech_breakdown
    df["Sell_HardStop"] = hard_stop
    
    # ========================================================
    # 技術評分 (綜合健康度)
    # ========================================================
    score = pd.Series(50, index=df.index)  # 基準分 50
    
    # 加分項
    score += np.where(df["SanYang"], 20, 0)           # 三陽開泰 +20
    score += np.where(df["SiHai"], 10, 0)             # 四海遊龍 再 +10
    score += np.where(is_trend_up, 10, 0)             # 趨勢向上 +10
    score += np.where(df["Close"] > df["MA240"], 5, 0)  # 站上年線 +5
    
    # 扣分項
    score -= np.where(df["Close"] < df["StopLine"], 20, 0)  # 跌破生命線 -20
    score -= np.where(sell_big_vol, 30, 0)                  # 爆量破位 -30
    score -= np.where(sell_profit, 10, 0)                   # 高檔過熱 -10
    score -= np.where(hard_stop, 40, 0)                     # 硬停損 -40
    
    df["Tech_Score"] = score.clip(0, 100)
    
    # ========================================================
    # 相容性欄位 (舊版程式可能會用到)
    # ========================================================
    df["Sell_Core"] = sell_big_vol
    df["Short_Signal"] = False  # 暫不支援做空
    
    return df
