import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from technical_indicators import add_indicators
import random

def get_valid_symbols():
    """
    قائمة موسعة بالرموز الصالحة للتداول
    """
    return {
        'GC=F': 'الذهب (XAU/USD)',
        'SI=F': 'الفضة (XAG/USD)',
        'CL=F': 'النفط الخام (WTI)',
        'NQ=F': 'ناسداك 100 (NDX100)',
        'ES=F': 'S&P 500 (SPX500)',
        'YM=F': 'داو جونز (DOW30)',
        '^GDAXI': 'داكس الألماني (GER40)',
        '^FTSE': 'فوتسي البريطاني (UK100)',
        '^N225': 'نيكاي الياباني (JP225)',
        'GBPUSD=X': 'جنيه/دولار (GBP/USD)',
        'EURUSD=X': 'يورو/دولار (EUR/USD)',
        'USDJPY=X': 'دولار/ين (USD/JPY)',
        'USDCHF=X': 'دولار/فرنك (USD/CHF)',
        'AUDUSD=X': 'استرالي/دولار (AUD/USD)',
        'USDCAD=X': 'دولار/كندي (USD/CAD)',
        'BTC-USD': 'بيتكوين/دولار (BTC/USD)',
        'ETH-USD': 'إيثيريوم/دولار (ETH/USD)',
        'BNB-USD': 'بينانس/دولار (BNB/USD)',
        'XRP-USD': 'ريبل/دولار (XRP/USD)',
        'ADA-USD': 'كاردانو/دولار (ADA/USD)'
    }

def _get_interval_and_range(period):
    """
    تحديد الفاصل الزمني وفترة التاريخ المناسبة
    """
    if period == '1m':
        interval = '1m'
        period_range = '1d'
    elif period == '1h':
        interval = '15m'
        period_range = '7d'
    elif period == '1d':
        interval = '1h'
        period_range = '7d'
    elif period == '1mo':
        interval = '1d'
        period_range = '1mo'
    elif period == '3mo':
        interval = '1d'
        period_range = '3mo'
    elif period == '6mo':
        interval = '1d'
        period_range = '6mo'
    elif period == '1y':
        interval = '1d'
        period_range = '1y'
    elif period == '2y':
        interval = '1d'
        period_range = '2y'
    else:
        interval = '1d'
        period_range = period
    
    return interval, period_range

def fetch_market_data(symbol, period='1m'):
    """
    تحسين جلب بيانات السوق من Yahoo Finance
    مع معالجة أفضل للأخطاء والحالات الاستثنائية
    """
    try:
        # الحصول على الفاصل الزمني والفترة المناسبين
        interval, period_range = _get_interval_and_range(period)
        
        # التحقق من صحة الرمز
        valid_symbols = get_valid_symbols()
        if symbol in valid_symbols:
            ticker = yf.Ticker(symbol)
        else:
            raise ValueError(f"الرمز {symbol} غير صالح. الرجاء اختيار رمز من القائمة المتاحة.")
        
        # جلب البيانات
        df = ticker.history(period=period_range, interval=interval)
        
        # التحقق من البيانات
        if df.empty:
            raise ValueError(f"لم يتم العثور على بيانات للرمز {symbol}")
        
        # معالجة البيانات المفقودة
        df = df.ffill()  # استخدام طريقة أحدث لتجنب التحذيرات
        
        # تعيين اسم للبيانات (سيكون مفيدًا لاحقًا)
        df.name = symbol
        
        return df

    except Exception as e:
        # إرجاع بيانات تجريبية في حالة فشل الاتصال بمصدر البيانات
        if "failed to fetch data" in str(e).lower() or "connection" in str(e).lower():
            print(f"تعذر الاتصال بمصدر البيانات. سيتم استخدام بيانات تجريبية: {str(e)}")
            return generate_sample_data(symbol, period)
        else:
            raise Exception(f"خطأ في جلب البيانات لـ {symbol}: {str(e)}")

def generate_sample_data(symbol, period):
    """
    إنشاء بيانات تجريبية في حالة فشل الاتصال
    ملاحظة: لا تستخدم هذه البيانات للتداول الحقيقي
    """
    current_time = datetime.now()
    
    # تحديد عدد الفترات بناءً على الإطار الزمني
    if period == '1m':
        num_periods = 390  # عدد دقائق جلسة التداول
        freq = 'min'
    elif period == '1h':
        num_periods = 24 * 7  # ساعات في أسبوع
        freq = 'H'
    elif period == '1d':
        num_periods = 30  # شهر من التداول
        freq = 'D'
    elif period == '1mo':
        num_periods = 30  # شهر
        freq = 'D'
    elif period == '3mo':
        num_periods = 90  # 3 أشهر
        freq = 'D'
    elif period == '6mo':
        num_periods = 180  # 6 أشهر
        freq = 'D'
    elif period == '1y':
        num_periods = 365  # سنة
        freq = 'D'
    else:
        num_periods = 100
        freq = 'D'
    
    # توليد فترات زمنية
    if freq == 'min':
        end_time = current_time
        start_time = end_time - timedelta(minutes=num_periods)
        date_range = pd.date_range(start=start_time, end=end_time, periods=num_periods)
    elif freq == 'H':
        end_time = current_time
        start_time = end_time - timedelta(hours=num_periods)
        date_range = pd.date_range(start=start_time, end=end_time, periods=num_periods)
    else:
        end_time = current_time
        start_time = end_time - timedelta(days=num_periods)
        date_range = pd.date_range(start=start_time, end=end_time, periods=num_periods)
    
    # تعيين سعر البداية بناءً على الرمز
    if 'GC=F' in symbol:
        start_price = 2000.0
    elif 'BTC' in symbol:
        start_price = 65000.0
    elif 'ETH' in symbol:
        start_price = 3500.0
    elif 'USD' in symbol:
        start_price = 1.2
    else:
        start_price = 100.0
    
    # توليد أسعار افتتاح مع تقلب عشوائي
    np.random.seed(42)  # للتكرارية
    volatility = 0.02
    if 'BTC' in symbol or 'ETH' in symbol:
        volatility = 0.04
    
    # توليد سلسلة أسعار مع اتجاه عام
    trend = np.random.choice([-1, 1]) * 0.001  # اتجاه صاعد أو هابط
    returns = np.random.normal(trend, volatility, num_periods)
    close_prices = start_price * (1 + np.cumsum(returns))
    
    # إضافة تقلبات يومية
    open_prices = close_prices * (1 + np.random.normal(0, 0.003, num_periods))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.005, num_periods)))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.005, num_periods)))
    
    # إنشاء حجم التداول
    volume = np.random.lognormal(10, 1, num_periods) * 1000
    
    # إنشاء إطار البيانات
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=date_range)
    
    # إضافة اسم للبيانات
    df.name = symbol
    
    return df

def prepare_data(df):
    """
    تحضير البيانات للتحليل مع إضافة مجموعة أوسع من المؤشرات الفنية
    """
    # نسخ البيانات لتجنب التعديل المباشر
    result_df = df.copy()
    
    # إضافة المؤشرات الفنية الأساسية
    indicators = [
        'SMA20', 'SMA50', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'ATR'
    ]
    
    result_df = add_indicators(result_df, indicators)
    
    # إضافة أعمدة لتخزين بيانات الموجة
    result_df['wave_count'] = 0
    result_df['wave_label'] = ''
    
    return result_df

def get_historical_data(symbol, period='1y', interval='1d'):
    """
    الحصول على بيانات تاريخية أكثر للتحليل وتدريب النماذج
    """
    try:
        # جلب البيانات
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # معالجة البيانات المفقودة
        df = df.dropna(subset=['Close'])
        
        # تخزين اسم الرمز في البيانات
        df.name = symbol
        
        return df
        
    except Exception as e:
        print(f"خطأ في جلب البيانات التاريخية: {str(e)}")
        return None

def get_correlated_assets(symbol):
    """
    البحث عن الأصول المرتبطة بالرمز المحدد
    """
    forex_correlations = {
        'EURUSD=X': ['GBPUSD=X', 'AUDUSD=X', 'USDCHF=X'],
        'GBPUSD=X': ['EURUSD=X', 'AUDUSD=X', 'USDCAD=X'],
        'USDJPY=X': ['USDCHF=X', 'AUDUSD=X', 'EURUSD=X'],
        'AUDUSD=X': ['EURUSD=X', 'GBPUSD=X', 'NZDUSD=X'],
        'USDCAD=X': ['USDCHF=X', 'GBPUSD=X', 'EURUSD=X'],
        'USDCHF=X': ['USDJPY=X', 'EURUSD=X', 'GBPUSD=X']
    }
    
    crypto_correlations = {
        'BTC-USD': ['ETH-USD', 'BNB-USD', 'ADA-USD'],
        'ETH-USD': ['BTC-USD', 'BNB-USD', 'XRP-USD'],
        'BNB-USD': ['BTC-USD', 'ETH-USD', 'ADA-USD'],
        'XRP-USD': ['ETH-USD', 'ADA-USD', 'BTC-USD'],
        'ADA-USD': ['BTC-USD', 'ETH-USD', 'BNB-USD']
    }
    
    index_correlations = {
        'ES=F': ['NQ=F', 'YM=F', '^FTSE'],
        'NQ=F': ['ES=F', 'YM=F', '^GDAXI'],
        'YM=F': ['ES=F', 'NQ=F', '^N225'],
        '^FTSE': ['ES=F', '^GDAXI', '^N225'],
        '^GDAXI': ['^FTSE', 'NQ=F', '^N225'],
        '^N225': ['^GDAXI', '^FTSE', 'YM=F']
    }
    
    commodity_correlations = {
        'GC=F': ['SI=F', 'XAUUSD=X', 'EURUSD=X'],
        'SI=F': ['GC=F', 'XAGUSD=X', 'AUDUSD=X'],
        'CL=F': ['USDCAD=X', 'BZ=F', 'NG=F']
    }
    
    # اختيار المجموعة المناسبة بناءً على نوع الرمز
    if symbol in forex_correlations:
        correlations = forex_correlations
    elif symbol in crypto_correlations:
        correlations = crypto_correlations
    elif symbol in index_correlations:
        correlations = index_correlations
    elif symbol in commodity_correlations:
        correlations = commodity_correlations
    else:
        return []
    
    # إرجاع الأصول المرتبطة
    return correlations.get(symbol, [])