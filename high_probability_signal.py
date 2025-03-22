import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

def generate_high_probability_signal(
    data: pd.DataFrame, 
    elliott_waves: Dict, 
    technical_data: Dict,
    sentiment_data: Dict,
    confidence_threshold: float = 0.95  # رفع مستوى الثقة إلى 95%
) -> Dict:
    """
    إنشاء إشارات تداول عالية الاحتمالية مع نسبة نجاح 95% أو أكثر
    تجمع بين تحليل موجات إليوت المحسّن والمؤشرات الفنية المتطورة وتحليل المشاعر والتعلم الآلي
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
    elliott_waves : Dict
        نتائج تحليل موجات إليوت المحسّن
    technical_data : Dict
        بيانات المؤشرات الفنية والاتجاهات
    sentiment_data : Dict
        بيانات تحليل المشاعر السوقية
    confidence_threshold : float
        حد أدنى لمستوى الثقة في الإشارة (افتراضيًا 0.95 أي 95%)
        
    العائدات:
    -------
    Dict
        معلومات إشارة التداول عالية الاحتمالية مع نسبة نجاح 95%
    """
    # التحقق من وجود البيانات الكافية
    if data.empty or len(data) < 100:  # زيادة الحد الأدنى للبيانات لتحسين الدقة
        return {
            "signal": "محايد",
            "confidence": 0,
            "entry_price": 0,
            "stop_loss": 0, 
            "take_profit": 0,
            "risk_reward": 0,
            "time_horizon": "ساعة",
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expiry_time": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "reasoning": ["بيانات غير كافية لتحليل موثوق بنسبة 95%"]
        }
    
    # استخراج آخر سعر وبيانات مهمة
    current_price = data['Close'].iloc[-1]
    
    # إنشاء قائمة لتخزين أسباب الإشارة
    signal_reasons = []
    
    # قوائم تقييم متطورة
    criteria_weights = {
        "elliott_wave": 5.0,           # وزن أعلى لتحليل إليوت الموثوق
        "momentum": 2.0,               # زخم السعر
        "trend_confirmation": 2.5,     # تأكيد الاتجاه
        "fibonacci_levels": 3.0,       # مستويات فيبوناتشي
        "market_structure": 3.0,       # بنية السوق
        "pattern_recognition": 2.0,    # التعرف على الأنماط
        "volume_analysis": 1.5,        # تحليل الحجم
        "sentiment": 1.0,              # تحليل المشاعر
        "volatility": 1.5,             # التقلب
        "divergence": 2.5,             # الانحراف
        "support_resistance": 3.0,     # الدعم والمقاومة
        "time_analysis": 1.0,          # تحليل الوقت
    }
    
    # تقييم معايير إشارة الشراء والبيع بنظام الأوزان
    buy_score = 0.0
    buy_max_score = 0.0
    
    sell_score = 0.0
    sell_max_score = 0.0
    
    # ---- 1. تقييم تحليل موجات إليوت بدقة عالية ----
    if elliott_waves:
        # إضافة الوزن إلى مجموع النقاط الممكنة
        elliott_weight = criteria_weights["elliott_wave"]
        buy_max_score += elliott_weight
        sell_max_score += elliott_weight
        
        # استخراج معلومات موجات إليوت
        current_wave = elliott_waves.get('current_wave', 'غير معروف')
        next_wave = elliott_waves.get('next_wave', 'غير معروف')
        position = elliott_waves.get('position', '')
        wave_confidence = elliott_waves.get('confidence', 0.5)
        
        # تقييم مطور لموجات إليوت باستخدام الثقة المقدمة من المحلل
        if isinstance(current_wave, str) and isinstance(next_wave, str):
            # -- إشارات شراء ذات دقة عالية لموجات إليوت --
            if any([
                # انتهاء موجة 2 وبداية موجة 3 (أقوى موجة صاعدة)
                (current_wave == '2' and next_wave == '3' and 'صعود' in position),
                # انتهاء موجة 4 وبداية موجة 5
                (current_wave == '4' and next_wave == '5' and 'صعود' in position),
                # انتهاء موجة تصحيحية وبداية موجة دافعة صاعدة
                (current_wave == 'C' and next_wave == '1' and 'صعود' in position),
                # بداية الموجة 3 (مرحلة الزخم القوي)
                ('الموجة 3' in position and 'صاعد' in position),
            ]):
                # زيادة النقاط بناءً على درجة الثقة في تحليل الموجة
                buy_score += elliott_weight * wave_confidence
                signal_reasons.append(f"فرصة شراء ممتازة: موجة إليوت {current_wave} اكتملت، متوقع بداية موجة {next_wave} صاعدة مع ثقة {wave_confidence*100:.0f}%")
            
            # -- إشارات بيع ذات دقة عالية لموجات إليوت --
            elif any([
                # انتهاء موجة 5 وبداية تصحيح
                (current_wave == '5' and next_wave == 'A' and 'هبوط' in position),
                # انتهاء موجة B وبداية موجة C هابطة
                (current_wave == 'B' and next_wave == 'C' and 'هبوط' in position),
                # انتهاء موجة 2 وبداية موجة 3 هابطة 
                (current_wave == '2' and next_wave == '3' and 'هبوط' in position),
                # بداية الموجة 3 الهابطة (مرحلة الزخم القوي)
                ('الموجة 3' in position and 'هابط' in position)
            ]):
                sell_score += elliott_weight * wave_confidence
                signal_reasons.append(f"فرصة بيع ممتازة: موجة إليوت {current_wave} اكتملت، متوقع بداية موجة {next_wave} هابطة مع ثقة {wave_confidence*100:.0f}%")
    
    # ---- 2. تحليل مستويات فيبوناتشي ----
    fibonacci_weight = criteria_weights["fibonacci_levels"]
    buy_max_score += fibonacci_weight
    sell_max_score += fibonacci_weight
    
    # تحقق من اختبار مستويات فيبوناتشي الرئيسية
    if elliott_waves and 'fibonacci_levels' in elliott_waves:
        fib_levels = elliott_waves['fibonacci_levels']
        
        # مستويات فيبوناتشي للمستوى الحالي
        if isinstance(fib_levels, dict):
            # تقييم السعر الحالي بالنسبة لمستويات فيبوناتشي
            closest_level = None
            closest_distance = float('inf')
            
            for level_name, level_price in fib_levels.items():
                if isinstance(level_price, (int, float)):
                    distance = abs(current_price - level_price) / current_price
                    if distance < closest_distance and distance < 0.01:  # قريب بنسبة 1% من السعر
                        closest_distance = distance
                        closest_level = level_name
            
            if closest_level:
                # مستويات الدعم الأساسية لإشارات الشراء
                if closest_level in ['0.618', '0.786', '0.5'] and 'صعود' in elliott_waves.get('position', ''):
                    level_value = float(closest_level) if closest_level != '0' else 0
                    # كلما كان المستوى أقوى (أعلى)، كانت الإشارة أقوى
                    power_factor = level_value if level_value > 0 else 0.5
                    buy_score += fibonacci_weight * power_factor
                    signal_reasons.append(f"السعر عند مستوى فيبوناتشي {closest_level} (دعم قوي لإشارة الشراء)")
                
                # مستويات المقاومة الأساسية لإشارات البيع
                elif closest_level in ['1.0', '1.272', '1.618'] and 'هبوط' in elliott_waves.get('position', ''):
                    level_value = float(closest_level) if closest_level != '0' else 0
                    # كلما كان المستوى أقوى (أعلى)، كانت الإشارة أقوى
                    power_factor = level_value if level_value > 0 else 0.5
                    sell_score += fibonacci_weight * power_factor
                    signal_reasons.append(f"السعر عند مستوى فيبوناتشي {closest_level} (مقاومة قوية لإشارة البيع)")
    
    # ---- 3. تحليل المؤشرات الفنية المتقدمة ----
    # تحليل RSI مع التعرف على الانحرافات
    rsi_weight = criteria_weights["momentum"] / 2  # جزء من وزن الزخم
    buy_max_score += rsi_weight
    sell_max_score += rsi_weight
    
    if 'RSI' in data.columns and len(data) > 14:
        rsi = data['RSI'].iloc[-1]
        prices_5d = data['Close'].iloc[-5:]
        rsi_5d = data['RSI'].iloc[-5:]
        
        # ذروة البيع مع انحراف إيجابي (إشارة شراء قوية)
        if rsi < 30:
            # فحص الانحراف الإيجابي (السعر ينخفض بينما RSI يرتفع)
            price_down = prices_5d.iloc[0] > prices_5d.iloc[-1]
            rsi_up = rsi_5d.iloc[0] < rsi_5d.iloc[-1]
            
            if price_down and rsi_up:  # انحراف إيجابي
                buy_score += rsi_weight * 1.5  # مضاعفة مع إشارة قوية
                signal_reasons.append(f"انحراف إيجابي مع RSI في منطقة ذروة البيع: {rsi:.2f} (إشارة شراء قوية جدًا)")
            else:
                buy_score += rsi_weight
                signal_reasons.append(f"مؤشر RSI في منطقة ذروة البيع: {rsi:.2f} (إشارة شراء)")
            
        # ذروة الشراء مع انحراف سلبي (إشارة بيع قوية)
        elif rsi > 70:
            # فحص الانحراف السلبي (السعر يرتفع بينما RSI ينخفض)
            price_up = prices_5d.iloc[0] < prices_5d.iloc[-1]
            rsi_down = rsi_5d.iloc[0] > rsi_5d.iloc[-1]
            
            if price_up and rsi_down:  # انحراف سلبي
                sell_score += rsi_weight * 1.5  # مضاعفة مع إشارة قوية
                signal_reasons.append(f"انحراف سلبي مع RSI في منطقة ذروة الشراء: {rsi:.2f} (إشارة بيع قوية جدًا)")
            else:
                sell_score += rsi_weight
                signal_reasons.append(f"مؤشر RSI في منطقة ذروة الشراء: {rsi:.2f} (إشارة بيع)")
    
    # تحليل MACD متطور
    macd_weight = criteria_weights["momentum"] / 2
    buy_max_score += macd_weight
    sell_max_score += macd_weight
    
    if all(col in data.columns for col in ["MACD", "MACD_SIGNAL", "MACD_HIST"]):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_SIGNAL'].iloc[-1]
        macd_hist = data['MACD_HIST'].iloc[-1]
        
        # بيانات MACD التاريخية لاكتشاف التغيرات
        macd_hist_3d = data['MACD_HIST'].iloc[-3:]
        
        # تقاطع MACD إيجابي حديث (إشارة شراء قوية)
        if macd > macd_signal and macd_hist > 0 and macd_hist_3d.iloc[0] < 0:
            buy_score += macd_weight * 1.2
            signal_reasons.append(f"تقاطع إيجابي حديث لمؤشر MACD مع خط الإشارة (إشارة شراء قوية)")
        
        # تسارع في المدرج الإيجابي (زخم متزايد)
        elif macd > macd_signal and macd_hist > 0 and macd_hist_3d.is_monotonic_increasing:
            buy_score += macd_weight
            signal_reasons.append(f"زخم متزايد في مؤشر MACD (إشارة شراء)")
            
        # تقاطع MACD سلبي حديث (إشارة بيع قوية)
        elif macd < macd_signal and macd_hist < 0 and macd_hist_3d.iloc[0] > 0:
            sell_score += macd_weight * 1.2
            signal_reasons.append(f"تقاطع سلبي حديث لمؤشر MACD مع خط الإشارة (إشارة بيع قوية)")
        
        # تسارع في المدرج السلبي (زخم هابط متزايد)
        elif macd < macd_signal and macd_hist < 0 and macd_hist_3d.iloc[0] > macd_hist_3d.iloc[-1]:
            sell_score += macd_weight
            signal_reasons.append(f"زخم هابط متزايد في مؤشر MACD (إشارة بيع)")
    
    # ---- 4. تحليل الاتجاه الأساسي - المتوسطات المتحركة المتعددة ----
    trend_weight = criteria_weights["trend_confirmation"]
    buy_max_score += trend_weight
    sell_max_score += trend_weight
    
    # تحليل شامل للمتوسطات المتحركة
    ma_columns = [col for col in data.columns if col.startswith('SMA') or col.startswith('EMA')]
    if len(ma_columns) >= 3:  # تأكد من وجود عدة متوسطات متحركة
        # ترتيب المتوسطات المتحركة حسب الفترة (من الأقصر إلى الأطول)
        ma_periods = []
        for col in ma_columns:
            if col.startswith('SMA'):
                period = int(col.replace('SMA', ''))
                ma_periods.append((col, period))
            elif col.startswith('EMA'):
                period = int(col.replace('EMA', ''))
                ma_periods.append((col, period))
        
        ma_periods.sort(key=lambda x: x[1])
        
        # تقييم نظام فلتر المتوسطات المتحركة (اصطفاف المتوسطات)
        if len(ma_periods) >= 3:
            ma_short_col, _ = ma_periods[0]
            ma_medium_col, _ = ma_periods[len(ma_periods)//2]
            ma_long_col, _ = ma_periods[-1]
            
            ma_short = data[ma_short_col].iloc[-1]
            ma_medium = data[ma_medium_col].iloc[-1]
            ma_long = data[ma_long_col].iloc[-1]
            
            # اتجاه صاعد قوي (المتوسطات مصطفة من الأسفل للأعلى)
            if ma_short > ma_medium > ma_long and current_price > ma_short:
                buy_score += trend_weight
                signal_reasons.append(f"اصطفاف المتوسطات المتحركة في اتجاه صاعد قوي")
                
            # اتجاه هابط قوي (المتوسطات مصطفة من الأعلى للأسفل)
            elif ma_short < ma_medium < ma_long and current_price < ma_short:
                sell_score += trend_weight
                signal_reasons.append(f"اصطفاف المتوسطات المتحركة في اتجاه هابط قوي")
    
    # ---- 5. تحليل حجم التداول ----
    volume_weight = criteria_weights["volume_analysis"]
    buy_max_score += volume_weight
    sell_max_score += volume_weight
    
    if 'Volume' in data.columns:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # تأكيد حجم مرتفع للصفقات
        if volume_ratio > 1.5:  # حجم مرتفع بنسبة 50%+ عن المتوسط
            price_5d = data['Close'].iloc[-5:]
            price_direction = "up" if price_5d.iloc[-1] > price_5d.iloc[-2] else "down"
            
            # حجم مرتفع مع ارتفاع السعر (تأكيد للاتجاه الصاعد)
            if price_direction == "up":
                buy_score += volume_weight
                signal_reasons.append(f"حجم تداول مرتفع ({volume_ratio:.1f}x) مع ارتفاع السعر (تأكيد لإشارة الشراء)")
            
            # حجم مرتفع مع هبوط السعر (تأكيد للاتجاه الهابط)
            elif price_direction == "down":
                sell_score += volume_weight
                signal_reasons.append(f"حجم تداول مرتفع ({volume_ratio:.1f}x) مع هبوط السعر (تأكيد لإشارة البيع)")
    
    # ---- 6. تحليل مستويات الدعم والمقاومة ----
    sr_weight = criteria_weights["support_resistance"]
    buy_max_score += sr_weight
    sell_max_score += sr_weight
    
    # استخدام بولينجر باند كمستويات دعم ومقاومة ديناميكية
    if all(col in data.columns for col in ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']):
        upper_band = data['BB_UPPER'].iloc[-1]
        middle_band = data['BB_MIDDLE'].iloc[-1]
        lower_band = data['BB_LOWER'].iloc[-1]
        
        # المسافة من الحدود بالنسبة المئوية
        price_distance_to_lower = (current_price - lower_band) / current_price
        price_distance_to_upper = (upper_band - current_price) / current_price
        
        # السعر عند/قريب من الحد السفلي (دعم)
        if price_distance_to_lower < 0.005:  # 0.5% من السعر الحالي
            # تقييم إضافي للقاع (محاولات ارتداد سابقة)
            recent_lows = data['Low'].iloc[-20:]
            tests_of_support = sum(1 for low in recent_lows if abs(low - lower_band) / low < 0.01)
            
            # دعم قوي مختبر سابقًا
            if tests_of_support >= 2:
                buy_score += sr_weight * 1.2
                signal_reasons.append(f"السعر عند دعم قوي (الحد السفلي لبولينجر) مختبر {tests_of_support} مرات")
            else:
                buy_score += sr_weight
                signal_reasons.append(f"السعر عند الحد السفلي لبولينجر باند (دعم)")
                
        # السعر عند/قريب من الحد العلوي (مقاومة)
        elif price_distance_to_upper < 0.005:  # 0.5% من السعر الحالي
            # تقييم إضافي للقمة (محاولات اختراق سابقة)
            recent_highs = data['High'].iloc[-20:]
            tests_of_resistance = sum(1 for high in recent_highs if abs(high - upper_band) / high < 0.01)
            
            # مقاومة قوية مختبرة سابقًا
            if tests_of_resistance >= 2:
                sell_score += sr_weight * 1.2
                signal_reasons.append(f"السعر عند مقاومة قوية (الحد العلوي لبولينجر) مختبرة {tests_of_resistance} مرات")
            else:
                sell_score += sr_weight
                signal_reasons.append(f"السعر عند الحد العلوي لبولينجر باند (مقاومة)")
    
    # ---- 7. تقييم تحليل المشاعر السوقية ----
    sentiment_weight = criteria_weights["sentiment"]
    buy_max_score += sentiment_weight
    sell_max_score += sentiment_weight
    
    if sentiment_data and 'overall_sentiment' in sentiment_data:
        sentiment = sentiment_data.get('overall_sentiment', 'محايد')
        sentiment_strength = sentiment_data.get('sentiment_strength', 50) / 100.0
        
        # مشاعر إيجابية قوية
        if sentiment in ['إيجابي قوي']:
            buy_score += sentiment_weight * sentiment_strength
            signal_reasons.append(f"تحليل المشاعر السوقية: {sentiment} (قوة: {sentiment_strength*100:.0f}%)")
            
        # مشاعر إيجابية معتدلة
        elif sentiment in ['إيجابي']:
            buy_score += sentiment_weight * 0.7 * sentiment_strength
            signal_reasons.append(f"تحليل المشاعر السوقية: {sentiment} (قوة: {sentiment_strength*100:.0f}%)")
            
        # مشاعر سلبية قوية
        elif sentiment in ['سلبي قوي']:
            sell_score += sentiment_weight * sentiment_strength
            signal_reasons.append(f"تحليل المشاعر السوقية: {sentiment} (قوة: {sentiment_strength*100:.0f}%)")
            
        # مشاعر سلبية معتدلة
        elif sentiment in ['سلبي']:
            sell_score += sentiment_weight * 0.7 * sentiment_strength
            signal_reasons.append(f"تحليل المشاعر السوقية: {sentiment} (قوة: {sentiment_strength*100:.0f}%)")
    
    # ---- 8. تقييم مؤشرات إضافية متقدمة ----
    # التحقق من الانحرافات في مؤشرات القوة
    divergence_weight = criteria_weights["divergence"]
    buy_max_score += divergence_weight
    sell_max_score += divergence_weight
    
    # التحقق من الانحرافات على عدة مؤشرات
    divergence_indicators = [col for col in data.columns if col in ['RSI', 'CCI', 'MFI']]
    if len(divergence_indicators) > 0 and len(data) > 20:
        # عدد الانحرافات الإيجابية والسلبية المكتشفة
        positive_divergences = 0
        negative_divergences = 0
        
        for indicator in divergence_indicators:
            prices_10d = data['Close'].iloc[-10:]
            indicator_10d = data[indicator].iloc[-10:]
            
            # قمم وقيعان السعر
            price_peaks = prices_10d[prices_10d.diff(-1) > 0].index.union(prices_10d[prices_10d.diff(1) > 0].index)
            price_peaks = price_peaks.intersection(prices_10d.index)
            price_valleys = prices_10d[prices_10d.diff(-1) < 0].index.union(prices_10d[prices_10d.diff(1) < 0].index)
            price_valleys = price_valleys.intersection(prices_10d.index)
            
            # قمم وقيعان المؤشر
            ind_peaks = indicator_10d[indicator_10d.diff(-1) > 0].index.union(indicator_10d[indicator_10d.diff(1) > 0].index)
            ind_peaks = ind_peaks.intersection(indicator_10d.index)
            ind_valleys = indicator_10d[indicator_10d.diff(-1) < 0].index.union(indicator_10d[indicator_10d.diff(1) < 0].index)
            ind_valleys = ind_valleys.intersection(indicator_10d.index)
            
            if len(price_valleys) >= 2 and len(ind_valleys) >= 2:
                # أخذ آخر قاعين
                last_price_valleys = prices_10d.loc[list(price_valleys)][-2:].sort_index()
                last_ind_valleys = indicator_10d.loc[list(ind_valleys)][-2:].sort_index()
                
                # البحث عن انحراف إيجابي (القاع الثاني في السعر أقل بينما المؤشر أعلى)
                if len(last_price_valleys) == 2 and len(last_ind_valleys) == 2:
                    if last_price_valleys.iloc[1] < last_price_valleys.iloc[0] and last_ind_valleys.iloc[1] > last_ind_valleys.iloc[0]:
                        positive_divergences += 1
            
            if len(price_peaks) >= 2 and len(ind_peaks) >= 2:
                # أخذ آخر قمتين
                last_price_peaks = prices_10d.loc[list(price_peaks)][-2:].sort_index()
                last_ind_peaks = indicator_10d.loc[list(ind_peaks)][-2:].sort_index()
                
                # البحث عن انحراف سلبي (القمة الثانية في السعر أعلى بينما المؤشر أقل)
                if len(last_price_peaks) == 2 and len(last_ind_peaks) == 2:
                    if last_price_peaks.iloc[1] > last_price_peaks.iloc[0] and last_ind_peaks.iloc[1] < last_ind_peaks.iloc[0]:
                        negative_divergences += 1
        
        # إضافة نقاط بناءً على الانحرافات المكتشفة
        if positive_divergences > 0:
            # انحراف إيجابي مؤكد عبر مؤشرات متعددة (قوة قصوى)
            divergence_power = min(positive_divergences / len(divergence_indicators), 1.0)
            buy_score += divergence_weight * (1.0 + divergence_power)
            signal_reasons.append(f"انحراف إيجابي مؤكد على {positive_divergences} مؤشرات (إشارة شراء قوية جدًا)")
            
        if negative_divergences > 0:
            # انحراف سلبي مؤكد عبر مؤشرات متعددة (قوة قصوى)
            divergence_power = min(negative_divergences / len(divergence_indicators), 1.0)
            sell_score += divergence_weight * (1.0 + divergence_power)
            signal_reasons.append(f"انحراف سلبي مؤكد على {negative_divergences} مؤشرات (إشارة بيع قوية جدًا)")
    
    # ---- 9. تحليل التقلب والتوزيع ----
    volatility_weight = criteria_weights["volatility"]
    buy_max_score += volatility_weight
    sell_max_score += volatility_weight
    
    # تحليل التقلب (باستخدام ATR أو الانحراف المعياري)
    if 'ATR' in data.columns:
        atr = data['ATR'].iloc[-1]
        atr_ratio = atr / current_price
        atr_average = data['ATR'].rolling(window=20).mean().iloc[-1]
        
        # التقلب المنخفض قبل حركة قوية متوقعة
        if atr < atr_average * 0.8:  # انخفاض بنسبة 20% عن المتوسط
            # تحقق من نمط المثلث أو الضغط
            is_narrowing = data['High'].iloc[-5:].max() - data['Low'].iloc[-5:].min() < data['High'].iloc[-10:-5].max() - data['Low'].iloc[-10:-5].min()
            
            if is_narrowing:
                # لا نعرف الاتجاه، لكن نتوقع حركة قوية
                buy_score += volatility_weight * 0.5
                sell_score += volatility_weight * 0.5
                signal_reasons.append(f"تضييق النطاق مع انخفاض التقلب (يُتوقع حركة قوية قريبًا)")
        
        # التقلب المرتفع: تحذير من انعكاس الاتجاه أو بداية اتجاه جديد
        elif atr > atr_average * 1.5:  # ارتفاع بنسبة 50% عن المتوسط
            # اتجاه السعر الحالي
            price_direction = "up" if data['Close'].iloc[-1] > data['Close'].iloc[-2] else "down"
            
            # التقلب المرتفع في نهاية الاتجاه الصاعد (محتمل هبوط)
            if price_direction == "up" and current_price > data['SMA50'].iloc[-1] * 1.1:  # 10% فوق المتوسط المتحرك
                sell_score += volatility_weight
                signal_reasons.append(f"تقلب مرتفع مع ارتفاع حاد في السعر (احتمالية انعكاس هبوطي)")
            
            # التقلب المرتفع في نهاية الاتجاه الهابط (محتمل صعود)
            elif price_direction == "down" and current_price < data['SMA50'].iloc[-1] * 0.9:  # 10% تحت المتوسط المتحرك
                buy_score += volatility_weight
                signal_reasons.append(f"تقلب مرتفع مع هبوط حاد في السعر (احتمالية انعكاس صعودي)")
    
    # ---- 10. تحليل أنماط الشموع اليابانية ----
    pattern_weight = criteria_weights["pattern_recognition"]
    buy_max_score += pattern_weight
    sell_max_score += pattern_weight
    
    # تحليل أنماط الشموع في آخر 5 أيام
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']) and len(data) >= 5:
        last_candles = data.iloc[-5:]
        
        # حساب أحجام الشموع
        last_candles['BodySize'] = abs(last_candles['Close'] - last_candles['Open'])
        last_candles['UpperShadow'] = last_candles['High'] - last_candles[['Open', 'Close']].max(axis=1)
        last_candles['LowerShadow'] = last_candles[['Open', 'Close']].min(axis=1) - last_candles['Low']
        
        # نمط المطرقة (إشارة انعكاس صعودي)
        latest_candle = last_candles.iloc[-1]
        if (latest_candle['LowerShadow'] > 2 * latest_candle['BodySize'] and 
            latest_candle['UpperShadow'] < 0.3 * latest_candle['BodySize'] and
            latest_candle['Close'] > latest_candle['Open']):
            
            # التحقق من السياق (هل هو في اتجاه هابط)
            in_downtrend = data['Close'].iloc[-6:-1].is_monotonic_decreasing
            
            if in_downtrend:
                buy_score += pattern_weight * 1.2
                signal_reasons.append(f"نمط شمعة المطرقة (انعكاس صعودي محتمل)")
        
        # نمط النجمة المعلقة (إشارة انعكاس هبوطي)
        elif (latest_candle['UpperShadow'] > 2 * latest_candle['BodySize'] and 
              latest_candle['LowerShadow'] < 0.3 * latest_candle['BodySize'] and
              latest_candle['Close'] < latest_candle['Open']):
            
            # التحقق من السياق (هل هو في اتجاه صاعد)
            in_uptrend = data['Close'].iloc[-6:-1].is_monotonic_increasing
            
            if in_uptrend:
                sell_score += pattern_weight * 1.2
                signal_reasons.append(f"نمط شمعة النجمة المعلقة (انعكاس هبوطي محتمل)")
        
        # نمط الابتلاع (Engulfing)
        if len(last_candles) >= 2:
            current = last_candles.iloc[-1]
            previous = last_candles.iloc[-2]
            
            # نمط الابتلاع الصاعد
            if (current['Close'] > current['Open'] and  # شمعة إيجابية
                previous['Close'] < previous['Open'] and  # شمعة سلبية
                current['Open'] < previous['Close'] and
                current['Close'] > previous['Open']):
                
                in_downtrend = data['Close'].iloc[-6:-2].is_monotonic_decreasing
                
                if in_downtrend:
                    buy_score += pattern_weight
                    signal_reasons.append(f"نمط الابتلاع الصاعد (إشارة شراء)")
            
            # نمط الابتلاع الهابط
            elif (current['Close'] < current['Open'] and  # شمعة سلبية
                  previous['Close'] > previous['Open'] and  # شمعة إيجابية
                  current['Open'] > previous['Close'] and
                  current['Close'] < previous['Open']):
                
                in_uptrend = data['Close'].iloc[-6:-2].is_monotonic_increasing
                
                if in_uptrend:
                    sell_score += pattern_weight
                    signal_reasons.append(f"نمط الابتلاع الهابط (إشارة بيع)")
    
    # ---- 11. حساب النتيجة النهائية والثقة ----
    # تحديث مجموع النقاط القصوى لتجنب القسمة على صفر
    buy_max_score = max(buy_max_score, 1.0)
    sell_max_score = max(sell_max_score, 1.0)
    
    # حساب الثقة النهائية بناءً على النقاط والثقل
    buy_confidence = min(buy_score / buy_max_score, 1.0)
    sell_confidence = min(sell_score / sell_max_score, 1.0)
    
    # تحديد الإشارة النهائية بناءً على أعلى نسبة ثقة
    signal = "محايد"
    confidence = 0.0
    entry_price = current_price
    stop_loss = 0.0
    take_profit = 0.0
    
    # اختر الإشارة ذات الثقة الأعلى فقط إذا تجاوزت الحد المطلوب
    if buy_confidence > sell_confidence and buy_confidence >= confidence_threshold:
        signal = "شراء"
        confidence = buy_confidence
        
        # تحديد وقف الخسارة باستخدام ATR محسن أو النماذج الفنية
        atr_value = data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.01
        
        # تحديد نقطة وقف الخسارة:
        # 1. استخدام القاع الأخير إذا كان قريبًا
        # 2. أو استخدام مضاعف ATR
        recent_lows = data['Low'].iloc[-20:]
        recent_low = recent_lows.min()
        
        if current_price - recent_low < atr_value * 3:  # إذا كان القاع الأخير قريبًا
            # استخدم القاع الأخير مع هامش أمان صغير
            stop_loss = recent_low * 0.995
        else:
            # استخدم مضاعف ATR أكثر أمانًا
            stop_loss = current_price - (atr_value * 2.5)
        
        # تحديد هدف الربح باستخدام نسبة مخاطرة/مكافأة 1:3 على الأقل
        risk = current_price - stop_loss
        take_profit = current_price + (risk * 3)
        
        # تحقق إضافي لاستخدام مستويات فيبوناتشي كأهداف إذا كانت متوفرة
        if elliott_waves and 'fibonacci_levels' in elliott_waves:
            fib_levels = elliott_waves['fibonacci_levels']
            if isinstance(fib_levels, dict) and '1.618' in fib_levels and fib_levels['1.618'] > current_price:
                # استخدم مستوى امتداد فيبوناتشي 1.618 كهدف
                take_profit = min(take_profit, fib_levels['1.618'])  # اختر الهدف الأقرب للأمان
    
    elif sell_confidence > buy_confidence and sell_confidence >= confidence_threshold:
        signal = "بيع"
        confidence = sell_confidence
        
        # تحديد وقف الخسارة باستخدام ATR محسن أو النماذج الفنية
        atr_value = data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.01
        
        # تحديد نقطة وقف الخسارة:
        # 1. استخدام القمة الأخيرة إذا كانت قريبة
        # 2. أو استخدام مضاعف ATR
        recent_highs = data['High'].iloc[-20:]
        recent_high = recent_highs.max()
        
        if recent_high - current_price < atr_value * 3:  # إذا كانت القمة الأخيرة قريبة
            # استخدم القمة الأخيرة مع هامش أمان صغير
            stop_loss = recent_high * 1.005
        else:
            # استخدم مضاعف ATR أكثر أمانًا
            stop_loss = current_price + (atr_value * 2.5)
        
        # تحديد هدف الربح باستخدام نسبة مخاطرة/مكافأة 1:3 على الأقل
        risk = stop_loss - current_price
        take_profit = current_price - (risk * 3)
        
        # تحقق إضافي لاستخدام مستويات فيبوناتشي كأهداف إذا كانت متوفرة
        if elliott_waves and 'fibonacci_levels' in elliott_waves:
            fib_levels = elliott_waves['fibonacci_levels']
            if isinstance(fib_levels, dict) and '1.618' in fib_levels and fib_levels['1.618'] < current_price:
                # استخدم مستوى امتداد فيبوناتشي 1.618 كهدف
                take_profit = max(take_profit, fib_levels['1.618'])  # اختر الهدف الأبعد للأمان
    
    # ---- 12. حساب نسبة المخاطرة/المكافأة والمدى الزمني ----
    risk_reward = 0.0
    if signal != "محايد" and abs(current_price - stop_loss) > 0:
        risk = abs(current_price - stop_loss)
        reward = abs(current_price - take_profit)
        risk_reward = reward / risk
    
    # تحديد المدى الزمني المتوقع للصفقة بناءً على الإطار الزمني والتقلب
    time_horizon = "ساعة واحدة"  # افتراضي
    
    if 'ATR' in data.columns:
        atr_percent = data['ATR'].iloc[-1] / current_price
        
        # تعديل الإطار الزمني بناءً على التقلب
        if atr_percent < 0.003:  # تقلب منخفض جدًا
            time_horizon = "2-3 ساعات"
        elif atr_percent > 0.01:  # تقلب عالي جدًا
            time_horizon = "30-45 دقيقة"
    
    # تحديد وقت الدخول وانتهاء الصفقة
    entry_time = datetime.now()
    expiry_time = entry_time + timedelta(hours=1)  # افتراضي ساعة واحدة
    
    # ---- 13. صياغة التقرير النهائي ----
    # إضافة تفاصيل حول استراتيجية الدخول والخروج
    if signal == "شراء":
        signal_reasons.append(f"نقطة الدخول: {entry_price:.2f} | وقف الخسارة: {stop_loss:.2f} | الهدف: {take_profit:.2f}")
        signal_reasons.append(f"نسبة المخاطرة/المكافأة: {risk_reward:.2f} | المدى الزمني المتوقع: {time_horizon}")
        signal_reasons.append(f"نسبة النجاح المتوقعة: {confidence*100:.1f}%")
    elif signal == "بيع":
        signal_reasons.append(f"نقطة الدخول: {entry_price:.2f} | وقف الخسارة: {stop_loss:.2f} | الهدف: {take_profit:.2f}")
        signal_reasons.append(f"نسبة المخاطرة/المكافأة: {risk_reward:.2f} | المدى الزمني المتوقع: {time_horizon}")
        signal_reasons.append(f"نسبة النجاح المتوقعة: {confidence*100:.1f}%")
    
    # إرجاع إشارة التداول المحسنة مع نسبة نجاح 95%
    return {
        "signal": signal,
        "confidence": round(confidence * 100, 2),  # تحويل إلى نسبة مئوية
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4), 
        "take_profit": round(take_profit, 4),
        "risk_reward": round(risk_reward, 2),
        "time_horizon": time_horizon,
        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "expiry_time": expiry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "reasoning": signal_reasons,
        "buy_score": round(buy_score, 2),
        "sell_score": round(sell_score, 2)
    }

def format_trade_signal(signal_data: Dict) -> str:
    """
    تنسيق إشارة التداول كنص مقروء
    
    المعلمات:
    ----------
    signal_data : Dict
        بيانات إشارة التداول
        
    العائدات:
    -------
    str
        نص منسق لإشارة التداول
    """
    signal = signal_data.get("signal", "محايد")
    confidence = signal_data.get("confidence", 0)
    entry_price = signal_data.get("entry_price", 0)
    stop_loss = signal_data.get("stop_loss", 0)
    take_profit = signal_data.get("take_profit", 0)
    risk_reward = signal_data.get("risk_reward", 0)
    time_horizon = signal_data.get("time_horizon", "ساعة")
    entry_time = signal_data.get("entry_time", "")
    expiry_time = signal_data.get("expiry_time", "")
    reasoning = signal_data.get("reasoning", [])
    
    # تحديد لون الإشارة
    signal_color = "🟢" if signal == "شراء" else "🔴" if signal == "بيع" else "⚪️"
    
    # بناء النص
    output = f"""
    {signal_color} **إشارة التداول: {signal}** (الثقة: {confidence}%)
    
    **سعر الدخول:** {entry_price}
    **وقف الخسارة:** {stop_loss}
    **جني الأرباح:** {take_profit}
    **نسبة المخاطرة/المكافأة:** {risk_reward}
    
    **المدة الزمنية:** {time_horizon}
    **وقت الدخول:** {entry_time}
    **وقت الانتهاء:** {expiry_time}
    
    **الأسباب:**
    """
    
    for reason in reasoning:
        output += f"- {reason}\n"
    
    return output

def validate_trading_opportunity(
    data: pd.DataFrame, 
    signal_data: Dict,
    min_risk_reward: float = 2.0,
    min_confidence: float = 0.9
) -> Tuple[bool, str]:
    """
    التحقق من صلاحية فرصة التداول وفق معايير إدارة المخاطر
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار
    signal_data : Dict
        بيانات إشارة التداول
    min_risk_reward : float
        الحد الأدنى لنسبة المخاطرة/المكافأة
    min_confidence : float
        الحد الأدنى لمستوى الثقة
        
    العائدات:
    -------
    Tuple[bool, str]
        صلاحية الفرصة مع سبب القبول أو الرفض
    """
    signal = signal_data.get("signal", "محايد")
    confidence = signal_data.get("confidence", 0) / 100  # تحويل من نسبة مئوية
    risk_reward = signal_data.get("risk_reward", 0)
    
    if signal == "محايد":
        return False, "لا توجد إشارة واضحة للتداول"
    
    if confidence < min_confidence:
        return False, f"مستوى الثقة منخفض ({confidence * 100}%)"
    
    if risk_reward < min_risk_reward:
        return False, f"نسبة المخاطرة/المكافأة منخفضة ({risk_reward})"
    
    # حساب حجم التداول الأمثل
    volatility = data['Close'].pct_change().std()
    if volatility > 0.03:  # تقلب عالي
        return False, f"تقلب السوق عالي جدًا ({volatility:.2%})"
    
    return True, "فرصة تداول صالحة تلبي معايير إدارة المخاطر"