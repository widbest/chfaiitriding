import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple

def timeframe_to_minutes(timeframe: str) -> int:
    """
    تحويل الإطار الزمني إلى دقائق
    """
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 1440  # 24 * 60
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 10080  # 7 * 24 * 60
    elif timeframe.endswith('mo'):
        return int(timeframe[:-2]) * 43200  # 30 * 24 * 60
    else:
        return int(timeframe)  # افتراض أن الرقم يمثل الدقائق

def format_number(value: Union[int, float]) -> str:
    """
    تنسيق الأرقام بشكل أفضل للعرض
    """
    if value is None:
        return "0"
    
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    
    # التعامل مع الأرقام الكبيرة
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    
    # التعامل مع الأرقام الصغيرة
    elif value >= 0.01 or value <= -0.01 or value == 0:
        return f"{value:.2f}"
    elif value >= 0.0001 or value <= -0.0001:
        return f"{value:.6f}"
    else:
        return f"{value:.8f}"

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> Tuple[float, float]:
    """
    حساب نسبة المخاطرة/المكافأة ومستوى جودة الصفقة
    """
    if entry_price == stop_loss:
        return 0, 0
    
    # حساب المخاطرة والمكافأة
    if entry_price > stop_loss:  # صفقة شراء
        risk = (entry_price - stop_loss) / entry_price
        reward = (take_profit - entry_price) / entry_price
    else:  # صفقة بيع
        risk = (stop_loss - entry_price) / entry_price
        reward = (entry_price - take_profit) / entry_price
    
    # حساب نسبة المخاطرة/المكافأة
    if risk == 0:
        ratio = 0
    else:
        ratio = reward / risk
    
    # تقييم جودة الصفقة
    if ratio >= 3:
        quality = 5  # ممتاز
    elif ratio >= 2:
        quality = 4  # جيد جدًا
    elif ratio >= 1.5:
        quality = 3  # جيد
    elif ratio >= 1:
        quality = 2  # مقبول
    else:
        quality = 1  # ضعيف
    
    return ratio, quality

def optimal_position_size(account_balance: float, risk_percentage: float, entry: float, stop_loss: float) -> float:
    """
    حساب الحجم الأمثل للمركز بناءً على إدارة المخاطر
    """
    # حساب المبلغ المراد المخاطرة به
    risk_amount = account_balance * (risk_percentage / 100)
    
    # حساب نسبة المخاطرة للصفقة
    risk_per_unit = abs(entry - stop_loss) / entry
    
    # حساب حجم المركز
    if risk_per_unit > 0:
        position_size = risk_amount / (entry * risk_per_unit)
    else:
        position_size = 0
    
    return position_size

def get_current_market_status() -> Dict:
    """
    الحصول على حالة السوق الحالية
    """
    now = datetime.now()
    
    # تحديد ما إذا كانت الأسواق الرئيسية مفتوحة أم مغلقة
    is_weekend = now.weekday() >= 5  # السبت أو الأحد
    
    # ساعات العمل المعتادة للأسواق المختلفة (بالتوقيت العالمي)
    # تم تبسيط المنطق للتوضيح
    hour = now.hour
    
    forex_open = not is_weekend or (hour >= 22 or hour < 22)  # سوق الفوركس يعمل معظم الوقت
    crypto_open = True  # العملات المشفرة تعمل على مدار الساعة
    us_market_open = not is_weekend and (13 <= hour < 20)  # سوق الأسهم الأمريكية
    eu_market_open = not is_weekend and (8 <= hour < 16)  # سوق الأسهم الأوروبية
    asia_market_open = not is_weekend and ((hour >= 0 and hour < 8) or hour >= 22)  # سوق الأسهم الآسيوية
    
    # صياغة الرسالة
    message = ""
    if now.hour >= 22 or now.hour < 5:
        trading_session = "الجلسة الآسيوية"
    elif now.hour >= 5 and now.hour < 12:
        trading_session = "الجلسة الأوروبية"
    elif now.hour >= 12 and now.hour < 20:
        trading_session = "الجلسة الأمريكية"
    else:
        trading_session = "تداخل الجلسات"
    
    # تحديد الأسواق النشطة حاليًا
    active_markets = []
    if forex_open:
        active_markets.append("الفوركس")
    if crypto_open:
        active_markets.append("العملات المشفرة")
    if us_market_open:
        active_markets.append("الأسهم الأمريكية")
    if eu_market_open:
        active_markets.append("الأسهم الأوروبية")
    if asia_market_open:
        active_markets.append("الأسهم الآسيوية")
    
    # تحديد الأحداث القادمة
    upcoming_events = []
    
    # يوم الجمعة بعد الظهر - التحضير لإغلاق نهاية الأسبوع
    if now.weekday() == 4 and now.hour >= 15:
        upcoming_events.append("إغلاق أسواق الفوركس لعطلة نهاية الأسبوع")
    
    # عطلة نهاية الأسبوع
    if is_weekend:
        upcoming_events.append("الأسواق التقليدية مغلقة لعطلة نهاية الأسبوع")
    
    # افتتاح الأسواق
    if now.weekday() == 6 and now.hour >= 22:
        upcoming_events.append("افتتاح سوق الفوركس للأسبوع الجديد")
    
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "trading_session": trading_session,
        "active_markets": active_markets,
        "upcoming_events": upcoming_events
    }

def recommend_best_timeframe(volatility: float, trading_style: str) -> str:
    """
    توصية بأفضل إطار زمني للتداول بناءً على التقلب وأسلوب التداول
    """
    # تصنيف مستوى التقلب
    if volatility < 0.5:
        volatility_level = "منخفض"
    elif volatility < 1.5:
        volatility_level = "متوسط"
    else:
        volatility_level = "مرتفع"
    
    # التوصية بناءً على أسلوب التداول ومستوى التقلب
    if trading_style == "scalping":
        if volatility_level == "مرتفع":
            return "1m"
        elif volatility_level == "متوسط":
            return "5m"
        else:
            return "15m"
    
    elif trading_style == "day_trading":
        if volatility_level == "مرتفع":
            return "15m"
        elif volatility_level == "متوسط":
            return "30m"
        else:
            return "1h"
    
    elif trading_style == "swing_trading":
        if volatility_level == "مرتفع":
            return "4h"
        elif volatility_level == "متوسط":
            return "1d"
        else:
            return "1d"
    
    else:  # position_trading
        if volatility_level == "مرتفع":
            return "1d"
        elif volatility_level == "متوسط":
            return "1w"
        else:
            return "1mo"

def calculate_correlation(asset1_changes: List[float], asset2_changes: List[float]) -> float:
    """
    حساب معامل الارتباط بين أصلين
    """
    if len(asset1_changes) != len(asset2_changes) or len(asset1_changes) < 2:
        return 0
    
    # حساب معامل الارتباط
    correlation = np.corrcoef(asset1_changes, asset2_changes)[0, 1]
    
    return correlation

def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%d %b %Y") -> str:
    """
    تنسيق التاريخ من صيغة إلى أخرى
    """
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError:
        return date_str

def is_market_open(symbol: str) -> Tuple[bool, str]:
    """
    التحقق مما إذا كان السوق مفتوحًا للرمز المحدد
    """
    now = datetime.now()
    weekday = now.weekday()  # 0 = الاثنين، 6 = الأحد
    hour = now.hour
    
    # العملات المشفرة مفتوحة دائمًا
    if 'BTC' in symbol or 'ETH' in symbol or 'BNB' in symbol or 'XRP' in symbol or '-USD' in symbol:
        return True, "مفتوح (24/7)"
    
    # أسواق الفوركس
    if 'USD' in symbol and ('JPY' in symbol or 'EUR' in symbol or 'GBP' in symbol or 'AUD' in symbol or 'CAD' in symbol or 'CHF' in symbol):
        # مغلق في عطلة نهاية الأسبوع
        if weekday >= 5:  # السبت والأحد
            return False, "مغلق (عطلة نهاية الأسبوع)"
        
        # مغلق في بعض الساعات يوم الجمعة
        if weekday == 4 and hour >= 22:  # الجمعة بعد 22:00
            return False, "مغلق (نهاية الأسبوع)"
        
        return True, "مفتوح"
    
    # الأسهم الأمريكية والمؤشرات
    if 'NQ=F' in symbol or 'ES=F' in symbol or 'YM=F' in symbol:
        # مغلق في عطلة نهاية الأسبوع
        if weekday >= 5:  # السبت والأحد
            return False, "مغلق (عطلة نهاية الأسبوع)"
        
        # ساعات التداول العادية (9:30 - 16:00 بتوقيت نيويورك، وهو 14:30 - 21:00 بالتوقيت العالمي)
        if 14 <= hour < 21:
            return True, "مفتوح (الجلسة العادية)"
        
        # ساعات التداول الممتدة
        if 9 <= hour < 14 or 21 <= hour < 23:
            return True, "مفتوح (الجلسة الممتدة)"
        
        return False, "مغلق"
    
    # الأسهم الأوروبية والمؤشرات
    if '^GDAXI' in symbol or '^FTSE' in symbol:
        # مغلق في عطلة نهاية الأسبوع
        if weekday >= 5:  # السبت والأحد
            return False, "مغلق (عطلة نهاية الأسبوع)"
        
        # ساعات التداول العادية (تقريبًا 8:00 - 16:30 بالتوقيت الأوروبي)
        if 8 <= hour < 17:
            return True, "مفتوح"
        
        return False, "مغلق"
    
    # السلع
    if 'GC=F' in symbol or 'SI=F' in symbol or 'CL=F' in symbol:
        # ساعات التداول المختلفة، مبسطة للتوضيح
        if weekday >= 5:  # السبت والأحد
            return False, "مغلق (عطلة نهاية الأسبوع)"
        
        return True, "مفتوح (قد يتغير حسب الجلسة)"
    
    # افتراضي
    return True, "حالة غير معروفة"

def get_market_trends(symbol: str) -> Dict:
    """
    تحليل اتجاهات السوق العامة بناءً على الرمز
    """
    # تحديد نوع الأصل
    if 'BTC' in symbol or 'ETH' in symbol or 'BNB' in symbol or 'XRP' in symbol or 'ADA' in symbol:
        asset_type = 'crypto'
    elif 'USD' in symbol and ('JPY' in symbol or 'EUR' in symbol or 'GBP' in symbol):
        asset_type = 'forex'
    elif 'GC=F' in symbol or 'SI=F' in symbol or 'CL=F' in symbol:
        asset_type = 'commodity'
    elif '^' in symbol or '=' in symbol:
        asset_type = 'index'
    else:
        asset_type = 'stock'
    
    # توليد اتجاهات عشوائية بناءً على نوع الأصل
    trends = {}
    
    if asset_type == 'crypto':
        trends['short_term'] = random.choice(['صاعد قوي', 'صاعد', 'متذبذب', 'هابط', 'هابط قوي'])
        trends['medium_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['long_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
    elif asset_type == 'forex':
        trends['short_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['medium_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['long_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
    elif asset_type == 'commodity':
        trends['short_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['medium_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['long_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
    else:  # stock or index
        trends['short_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['medium_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
        trends['long_term'] = random.choice(['صاعد', 'متذبذب', 'هابط'])
    
    # توليد قوة الاتجاه
    strength = {
        'short_term': random.randint(50, 95),
        'medium_term': random.randint(40, 90),
        'long_term': random.randint(30, 85)
    }
    
    return {
        'symbol': symbol,
        'trends': trends,
        'strength': strength,
        'overall_trend': random.choice(['صاعد', 'متذبذب', 'هابط']),
        'volatility': random.choice(['منخفضة', 'متوسطة', 'مرتفعة']),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }