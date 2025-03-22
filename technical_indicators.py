import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

def get_available_indicators() -> Dict[str, str]:
    """
    الحصول على قائمة بالمؤشرات الفنية المتاحة
    """
    return {
        "SMA20": "المتوسط المتحرك البسيط (20)",
        "SMA50": "المتوسط المتحرك البسيط (50)",
        "SMA200": "المتوسط المتحرك البسيط (200)",
        "EMA10": "المتوسط المتحرك الأسي (10)",
        "EMA20": "المتوسط المتحرك الأسي (20)",
        "RSI": "مؤشر القوة النسبية",
        "MACD": "تقارب وتباعد المتوسطات المتحركة",
        "BB_UPPER": "بولينجر باند (الحد العلوي)",
        "BB_LOWER": "بولينجر باند (الحد السفلي)",
        "ATR": "متوسط المدى الحقيقي",
        "CCI": "مؤشر قناة السلع",
        "STOCH_K": "ستوكاستك %K",
        "STOCH_D": "ستوكاستك %D",
        "OBV": "توازن الحجم",
        "PSAR": "القطع المكافئ",
        "ADX": "مؤشر الاتجاه المتوسط",
        "AROON_UP": "مؤشر آرون (صعود)",
        "AROON_DOWN": "مؤشر آرون (هبوط)",
        "ICHIMOKU_CONVERSION": "خط تحويل إيشيموكو",
        "ICHIMOKU_BASE": "خط أساس إيشيموكو",
        "VWAP": "متوسط السعر المرجح بالحجم"
    }

def add_indicators(df: pd.DataFrame, indicators_list: List[str]) -> pd.DataFrame:
    """
    إضافة المؤشرات الفنية المطلوبة إلى إطار البيانات
    """
    # نسخ البيانات لتجنب التعديل المباشر
    result_df = df.copy()
    
    # التأكد من وجود عمود الحجم
    if 'Volume' not in result_df.columns:
        result_df['Volume'] = 0
    
    # إضافة المؤشرات المطلوبة
    for indicator in indicators_list:
        if indicator in result_df.columns:
            continue  # تجاوز المؤشرات الموجودة بالفعل
            
        if indicator == "SMA20":
            result_df['SMA20'] = calculate_sma(result_df['Close'], 20)
        
        elif indicator == "SMA50":
            result_df['SMA50'] = calculate_sma(result_df['Close'], 50)
        
        elif indicator == "SMA200":
            result_df['SMA200'] = calculate_sma(result_df['Close'], 200)
        
        elif indicator == "EMA10":
            result_df['EMA10'] = calculate_ema(result_df['Close'], 10)
        
        elif indicator == "EMA20":
            result_df['EMA20'] = calculate_ema(result_df['Close'], 20)
        
        elif indicator == "RSI":
            result_df['RSI'] = calculate_rsi(result_df['Close'], 14)
        
        elif indicator == "MACD":
            macd_data = calculate_macd(result_df['Close'])
            result_df['MACD'] = macd_data['MACD']
            result_df['MACD_SIGNAL'] = macd_data['SIGNAL']
            result_df['MACD_HIST'] = macd_data['HIST']
        
        elif indicator == "BB_UPPER" or indicator == "BB_LOWER" or indicator == "BB_MIDDLE":
            bb_data = calculate_bollinger_bands(result_df['Close'], 20, 2)
            result_df['BB_UPPER'] = bb_data['UPPER']
            result_df['BB_MIDDLE'] = bb_data['MIDDLE']
            result_df['BB_LOWER'] = bb_data['LOWER']
        
        elif indicator == "ATR":
            result_df['ATR'] = calculate_atr(result_df['High'], result_df['Low'], result_df['Close'], 14)
        
        elif indicator == "CCI":
            result_df['CCI'] = calculate_cci(result_df['High'], result_df['Low'], result_df['Close'], 20)
        
        elif indicator == "STOCH_K" or indicator == "STOCH_D":
            stoch_data = calculate_stochastic(result_df['High'], result_df['Low'], result_df['Close'], 14, 3)
            result_df['STOCH_K'] = stoch_data['K']
            result_df['STOCH_D'] = stoch_data['D']
        
        elif indicator == "OBV":
            result_df['OBV'] = calculate_obv(result_df['Close'], result_df['Volume'])
        
        elif indicator == "PSAR":
            result_df['PSAR'] = calculate_parabolic_sar(result_df['High'], result_df['Low'], result_df['Close'])
        
        elif indicator == "ADX":
            result_df['ADX'] = calculate_adx(result_df['High'], result_df['Low'], result_df['Close'], 14)
        
        elif indicator in ["AROON_UP", "AROON_DOWN"]:
            aroon_data = calculate_aroon(result_df['High'], result_df['Low'], 14)
            result_df['AROON_UP'] = aroon_data['AROON_UP']
            result_df['AROON_DOWN'] = aroon_data['AROON_DOWN']
        
        elif indicator in ["ICHIMOKU_CONVERSION", "ICHIMOKU_BASE"]:
            ichimoku_data = calculate_ichimoku(result_df['High'], result_df['Low'], result_df['Close'])
            result_df['ICHIMOKU_CONVERSION'] = ichimoku_data['CONVERSION']
            result_df['ICHIMOKU_BASE'] = ichimoku_data['BASE']
            result_df['ICHIMOKU_SPAN_A'] = ichimoku_data['SPAN_A']
            result_df['ICHIMOKU_SPAN_B'] = ichimoku_data['SPAN_B']
        
        elif indicator == "VWAP":
            result_df['VWAP'] = calculate_vwap(result_df['High'], result_df['Low'], result_df['Close'], result_df['Volume'])
    
    return result_df

def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    حساب المتوسط المتحرك البسيط
    """
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    حساب المتوسط المتحرك الأسي
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    حساب مؤشر القوة النسبية (RSI)
    """
    # حساب التغيرات
    delta = data.diff()
    
    # تقسيم التغيرات إلى إيجابية وسلبية
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # حساب المتوسط المتحرك الأسي للربح والخسارة
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # حساب القوة النسبية
    rs = avg_gain / (avg_loss + 1e-10)  # تجنب القسمة على صفر
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    حساب مؤشر تقارب وتباعد المتوسطات المتحركة (MACD)
    """
    # حساب المتوسطات المتحركة الأسية
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    # حساب MACD
    macd_line = ema_fast - ema_slow
    
    # حساب خط الإشارة
    signal_line = calculate_ema(macd_line, signal_period)
    
    # حساب الرسم البياني للفرق
    histogram = macd_line - signal_line
    
    return {
        'MACD': macd_line,
        'SIGNAL': signal_line,
        'HIST': histogram
    }

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    حساب حزمة بولينجر (Bollinger Bands)
    """
    # حساب المتوسط المتحرك
    middle_band = calculate_sma(data, period)
    
    # حساب الانحراف المعياري
    std = data.rolling(window=period).std()
    
    # حساب الحزام العلوي والسفلي
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return {
        'UPPER': upper_band,
        'MIDDLE': middle_band,
        'LOWER': lower_band
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    حساب متوسط المدى الحقيقي (ATR)
    """
    # حساب المدى الحقيقي
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # حساب المتوسط المتحرك الأسي للمدى الحقيقي
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    حساب مؤشر قناة السلع (CCI)
    """
    # حساب السعر النموذجي
    typical_price = (high + low + close) / 3
    
    # حساب المتوسط المتحرك للسعر النموذجي
    ma_tp = typical_price.rolling(window=period).mean()
    
    # حساب الانحراف المعياري
    md = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
    
    # حساب مؤشر قناة السلع
    cci = (typical_price - ma_tp) / (0.015 * md)
    
    return cci

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    حساب مؤشر ستوكاستك (Stochastic)
    """
    # حساب القيم القصوى والدنيا خلال الفترة
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    # حساب %K
    k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))  # تجنب القسمة على صفر
    
    # حساب %D
    d = k.rolling(window=d_period).mean()
    
    return {
        'K': k,
        'D': d
    }

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    حساب مؤشر توازن الحجم (OBV)
    """
    # حساب تغير السعر
    price_change = close.diff()
    
    # إنشاء سلسلة الوزن
    obv_values = pd.Series(0, index=close.index)
    
    # حساب OBV
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] - volume.iloc[i]
        else:
            obv_values.iloc[i] = obv_values.iloc[i-1]
    
    return obv_values

def calculate_parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    حساب مؤشر القطع المكافئ (Parabolic SAR)
    """
    # إعداد القيم الأولية
    psar = pd.Series(0.0, index=close.index)
    psar.iloc[0] = low.iloc[0]  # بدء الاتجاه الصاعد
    
    # إعداد القيم المتغيرة
    trend = 1  # 1 للصاعد، -1 للهابط
    af = af_start  # معامل التسارع
    extreme_point = high.iloc[0]  # النقطة القصوى للاتجاه الصاعد
    
    # حساب المؤشر
    for i in range(1, len(close)):
        # تحديث القيمة
        psar.iloc[i] = psar.iloc[i-1] + af * (extreme_point - psar.iloc[i-1])
        
        # التحقق من تغير الاتجاه
        if trend == 1:  # اتجاه صاعد
            # تحديث القيمة لضمان أن SAR أقل من القيمة الدنيا للشمعة السابقة والحالية
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i])
            
            # تحديث النقطة القصوى
            if high.iloc[i] > extreme_point:
                extreme_point = high.iloc[i]
                af = min(af + af_increment, af_max)
            
            # التحقق من عكس الاتجاه
            if psar.iloc[i] > low.iloc[i]:
                trend = -1
                psar.iloc[i] = extreme_point
                extreme_point = low.iloc[i]
                af = af_start
        
        else:  # اتجاه هابط
            # تحديث القيمة لضمان أن SAR أعلى من القيمة القصوى للشمعة السابقة والحالية
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i])
            
            # تحديث النقطة القصوى
            if low.iloc[i] < extreme_point:
                extreme_point = low.iloc[i]
                af = min(af + af_increment, af_max)
            
            # التحقق من عكس الاتجاه
            if psar.iloc[i] < high.iloc[i]:
                trend = 1
                psar.iloc[i] = extreme_point
                extreme_point = high.iloc[i]
                af = af_start
    
    return psar

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    حساب مؤشر الاتجاه المتوسط (ADX)
    """
    # حساب +DM و -DM
    plus_dm = high.diff()
    minus_dm = low.diff(-1).abs()
    
    # قيم +DM و -DM حيث +DM > -DM و > 0
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    
    # قيم -DM حيث -DM > +DM و > 0
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # حساب ATR
    atr = calculate_atr(high, low, close, period)
    
    # حساب +DI و -DI
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # حساب DX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    
    # حساب ADX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    حساب مؤشر آرون (Aroon)
    """
    # حساب آرون للصعود
    rolling_high = high.rolling(window=period)
    aroon_up = 100 * (period - rolling_high.apply(lambda x: x.argmax(), raw=True)) / period
    
    # حساب آرون للهبوط
    rolling_low = low.rolling(window=period)
    aroon_down = 100 * (period - rolling_low.apply(lambda x: x.argmin(), raw=True)) / period
    
    return {
        'AROON_UP': aroon_up,
        'AROON_DOWN': aroon_down
    }

def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    """
    حساب مؤشر إيشيموكو (Ichimoku)
    """
    # تحديد الفترات
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # حساب Tenkan-sen (خط التحويل)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # حساب Kijun-sen (خط الأساس)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # حساب Senkou Span A (السحابة المتقدمة A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # حساب Senkou Span B (السحابة المتقدمة B)
    senkou_high = high.rolling(window=senkou_span_b_period).max()
    senkou_low = low.rolling(window=senkou_span_b_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
    
    # حساب Chikou Span (الخط المتأخر)
    chikou_span = close.shift(-kijun_period)
    
    return {
        'CONVERSION': tenkan_sen,
        'BASE': kijun_sen,
        'SPAN_A': senkou_span_a,
        'SPAN_B': senkou_span_b,
        'LAGGING': chikou_span
    }

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    حساب متوسط السعر المرجح بالحجم (VWAP)
    """
    # حساب السعر النموذجي
    typical_price = (high + low + close) / 3
    
    # ضرب السعر بالحجم
    dollar_volume = typical_price * volume
    
    # حساب المجموع التراكمي
    cum_dollar_volume = dollar_volume.cumsum()
    cum_volume = volume.cumsum()
    
    # حساب VWAP
    vwap = cum_dollar_volume / cum_volume
    
    return vwap

def identify_chart_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    تحديد أنماط الرسم البياني (مثل الرأس والكتفين، المثلثات، إلخ)
    """
    patterns = []
    
    # التحقق من وجود نمط الرأس والكتفين
    head_and_shoulders = find_head_and_shoulders(df)
    if head_and_shoulders:
        patterns.append(head_and_shoulders)
    
    # التحقق من وجود نمط المثلث
    triangle = find_triangle(df)
    if triangle:
        patterns.append(triangle)
    
    # التحقق من وجود نمط الفنجان والمقبض
    cup_and_handle = find_cup_and_handle(df)
    if cup_and_handle:
        patterns.append(cup_and_handle)
    
    return patterns

def find_head_and_shoulders(df: pd.DataFrame) -> Optional[Dict]:
    """
    البحث عن نمط الرأس والكتفين
    """
    # حساب القمم والقيعان (التنفيذ الفعلي سيكون أكثر تعقيدًا)
    # هذه مجرد نسخة مبسطة للتوضيح
    
    return None  # تنفيذ مستقبلي

def find_triangle(df: pd.DataFrame) -> Optional[Dict]:
    """
    البحث عن نمط المثلث
    """
    # تنفيذ مستقبلي
    return None

def find_cup_and_handle(df: pd.DataFrame) -> Optional[Dict]:
    """
    البحث عن نمط الفنجان والمقبض
    """
    # تنفيذ مستقبلي
    return None