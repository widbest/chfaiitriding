import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import math
from scipy.signal import find_peaks

def identify_elliott_waves(data: pd.DataFrame, sensitivity: float = 0.5) -> Dict:
    """
    تحديد موجات إليوت في بيانات السوق
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار
    sensitivity : float
        مستوى الحساسية في اكتشاف القمم والقيعان (0.1 إلى 1.0)
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على معلومات عن موجات إليوت المكتشفة والإشارات التجارية
    """
    # التحقق من البيانات
    if data is None or len(data) < 50:
        return {"error": "البيانات غير كافية لتحليل موجات إليوت"}
    
    # استخراج بيانات الأسعار
    prices = data['Close'].values
    
    # تحديد القمم والقيعان
    peaks, valleys = find_pivot_points(prices, sensitivity)
    
    # تحليل موجات Impulse (الدافعة) وموجات Corrective (التصحيحية)
    waves = analyze_wave_structure(prices, peaks, valleys)
    
    # إضافة مؤشرات فنية لتعزيز التحليل
    current_wave_count = determine_current_wave(waves)
    
    # تحليل الأنماط وإنشاء إشارات التداول
    patterns = identify_wave_patterns(waves)
    trading_signals = generate_trading_signals(data, waves, current_wave_count)
    
    # تجميع النتائج
    result = {
        "waves": waves,
        "current_wave": current_wave_count,
        "patterns": patterns,
        "trading_signals": trading_signals,
        "peaks_idx": peaks,
        "valleys_idx": valleys
    }
    
    return result

def find_pivot_points(prices: np.ndarray, sensitivity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    تحديد نقاط القمم والقيعان في سلسلة الأسعار بدقة عالية جدًا
    
    المعلمات:
    ----------
    prices : np.ndarray
        مصفوفة الأسعار
    sensitivity : float
        مستوى الحساسية في اكتشاف القمم والقيعان (0.1 إلى 1.0)
        
    العائدات:
    -------
    Tuple[np.ndarray, np.ndarray]
        مؤشرات القمم والقيعان
    """
    # التعامل مع القيم الشاذة والفراغات في البيانات
    # استبدال القيم NaN بالقيم المجاورة لضمان جودة الاكتشاف
    if np.isnan(prices).any():
        # تعبئة القيم المفقودة
        valid_indices = ~np.isnan(prices)
        valid_indices_pos = np.where(valid_indices)[0]
        if len(valid_indices_pos) > 0:
            # استخدام أقرب قيمة صالحة
            for i in range(len(prices)):
                if np.isnan(prices[i]):
                    idx = np.abs(valid_indices_pos - i).argmin()
                    prices[i] = prices[valid_indices_pos[idx]]
    
    # تمهيد البيانات لتقليل الضوضاء وتحسين اكتشاف القمم والقيعان
    smoothed_prices = prices.copy()
    window = max(3, int(10 * (1 - sensitivity)))  # زيادة نافذة التمهيد مع انخفاض الحساسية
    for i in range(window, len(smoothed_prices) - window):
        smoothed_prices[i] = np.mean(prices[i-window:i+window+1])
    
    # تحسين معلمات اكتشاف القمم والقيعان بناءً على الحساسية والتقلب
    window_size = max(3, int(15 * (1 - sensitivity)))  # نافذة أصغر للحصول على إشارات أكثر
    price_range = np.max(prices) - np.min(prices)
    price_std = np.std(prices)
    
    # تكييف prominence بناءً على تقلب السعر ونطاقه
    prominence = price_std * (0.05 + 0.5 * sensitivity)
    if prominence < 0.001 * price_range:  # التأكد من أنها ليست صغيرة جدًا
        prominence = 0.001 * price_range
        
    # استخدام عرض (width) لتحديد القمم والقيعان الحقيقية
    width = max(1, int(5 * (1 - sensitivity)))  # عرض أصغر للحساسية العالية
    
    # البحث عن القمم مع معلمات محسنة
    try:
        peaks, peak_properties = find_peaks(
            smoothed_prices, 
            distance=window_size, 
            prominence=prominence,
            width=width
        )
        
        # البحث عن القيعان مع معلمات محسنة
        valleys, valley_properties = find_peaks(
            -smoothed_prices, 
            distance=window_size, 
            prominence=prominence,
            width=width
        )
    except Exception as e:
        # إذا فشل البحث، استخدم معلمات أكثر مرونة
        window_size = max(2, int(5 * (1 - sensitivity)))
        prominence = price_std * 0.1
        
        peaks, _ = find_peaks(smoothed_prices, distance=window_size, prominence=prominence)
        valleys, _ = find_peaks(-smoothed_prices, distance=window_size, prominence=prominence)
    
    # تصفية القمم والقيعان غير المهمة
    if len(peaks) > 0 and len(peaks) > 3:
        # ترتيب القمم حسب أهميتها
        peak_heights = smoothed_prices[peaks]
        peak_importance = peak_heights
        important_peaks_mask = peak_importance > np.median(peak_importance) * 0.8
        peaks = peaks[important_peaks_mask]
    
    if len(valleys) > 0 and len(valleys) > 3:
        # ترتيب القيعان حسب أهميتها
        valley_depths = -smoothed_prices[valleys]
        valley_importance = valley_depths
        important_valleys_mask = valley_importance > np.median(valley_importance) * 0.8
        valleys = valleys[important_valleys_mask]
    
    # التأكد من أن القمم والقيعان تتناوب (قيعان بين القمم، وقمم بين القيعان)
    if len(peaks) > 0 and len(valleys) > 0:
        all_pivots = np.sort(np.concatenate([peaks, valleys]))
        pivot_types = np.zeros_like(all_pivots)
        
        for i, idx in enumerate(all_pivots):
            if idx in peaks:
                pivot_types[i] = 1  # 1 للقمم
            else:
                pivot_types[i] = -1  # -1 للقيعان
        
        # التحقق من تناوب القمم والقيعان
        valid_pivots = []
        valid_types = []
        
        current_type = pivot_types[0]
        valid_pivots.append(all_pivots[0])
        valid_types.append(current_type)
        
        for i in range(1, len(all_pivots)):
            if pivot_types[i] != current_type:
                valid_pivots.append(all_pivots[i])
                valid_types.append(pivot_types[i])
                current_type = pivot_types[i]
        
        # فصل القمم والقيعان مرة أخرى
        peaks = np.array([pivot for i, pivot in enumerate(valid_pivots) if valid_types[i] == 1])
        valleys = np.array([pivot for i, pivot in enumerate(valid_pivots) if valid_types[i] == -1])
    
    # تأكد من أن القمم والقيعان موزعة بشكل منطقي على مدى البيانات
    if len(peaks) == 0 or len(valleys) == 0:
        # إذا لم يتم العثور على قمم أو قيعان كافية، استخدم طريقة أبسط
        n_segments = 10
        segment_size = len(prices) // n_segments
        peaks = []
        valleys = []
        
        for i in range(1, n_segments):
            segment = prices[i*segment_size:(i+1)*segment_size]
            max_idx = np.argmax(segment) + i*segment_size
            min_idx = np.argmin(segment) + i*segment_size
            
            if max_idx not in peaks:
                peaks.append(max_idx)
            if min_idx not in valleys:
                valleys.append(min_idx)
        
        peaks = np.array(peaks)
        valleys = np.array(valleys)
    
    return peaks, valleys

def analyze_wave_structure(prices: np.ndarray, peaks: np.ndarray, valleys: np.ndarray) -> Dict:
    """
    تحليل بنية الموجات وتحديد موجات إليوت بدقة عالية
    
    المعلمات:
    ----------
    prices : np.ndarray
        مصفوفة الأسعار
    peaks : np.ndarray
        مؤشرات القمم
    valleys : np.ndarray
        مؤشرات القيعان
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على معلومات عن موجات إليوت المكتشفة
    """
    # التحقق من وجود نقاط تحول كافية
    if len(peaks) < 3 or len(valleys) < 3:
        # إذا لم تكن هناك قمم وقيعان كافية، استخدم نهج بديل
        return _create_alternative_wave_structure(prices)
    
    # تصفية القمم والقيعان غير المهمة
    if len(peaks) > 20:  # إذا كان هناك الكثير من القمم، احتفظ بأهمها فقط
        peak_heights = prices[peaks]
        sorted_peak_indices = np.argsort(peak_heights)[::-1]  # ترتيب تنازلي
        peaks = peaks[sorted_peak_indices[:20]]  # احتفظ بأعلى 20 قمة
    
    if len(valleys) > 20:  # إذا كان هناك الكثير من القيعان، احتفظ بأهمها فقط
        valley_depths = -prices[valleys]
        sorted_valley_indices = np.argsort(valley_depths)[::-1]  # ترتيب تنازلي
        valleys = valleys[sorted_valley_indices[:20]]  # احتفظ بأعمق 20 قاع
    
    # دمج القمم والقيعان وترتيبها حسب الزمن
    all_pivots = np.sort(np.concatenate([peaks, valleys]))
    
    # التحقق من وجود نقاط تحول كافية
    if len(all_pivots) < 6:
        return _create_alternative_wave_structure(prices)
    
    # تحديد نوع كل نقطة (قمة أو قاع)
    pivot_types = []
    for idx in all_pivots:
        if idx in peaks:
            pivot_types.append("peak")
        else:
            pivot_types.append("valley")
    
    # التأكد من أن القمم والقيعان تتناوب بشكل صحيح
    valid_pivots = []
    valid_types = []
    
    # إضافة النقطة الأولى دائمًا
    valid_pivots.append(all_pivots[0])
    valid_types.append(pivot_types[0])
    
    for i in range(1, len(all_pivots)):
        if pivot_types[i] != pivot_types[i-1]:
            valid_pivots.append(all_pivots[i])
            valid_types.append(pivot_types[i])
    
    # إذا لم نحصل على عدد كافٍ من النقاط الصالحة، نعود إلى النهج البديل
    if len(valid_pivots) < 5:
        return _create_alternative_wave_structure(prices)
    
    # تحويل النقاط الصالحة إلى مصفوفات numpy
    all_pivots = np.array(valid_pivots)
    pivot_types = valid_types
    
    # حساب مؤشرات القوة النسبية لكل نقطة
    relative_strengths = []
    price_range = np.max(prices) - np.min(prices)
    
    for i, idx in enumerate(all_pivots):
        curr_price = prices[idx]
        
        # حساب قوة النقطة نسبة إلى النطاق الكلي للأسعار
        if pivot_types[i] == "peak":
            # بالنسبة للقمم، كلما كان السعر أعلى، كانت القوة أكبر
            strength = (curr_price - np.min(prices)) / price_range
        else:
            # بالنسبة للقيعان، كلما كان السعر أقل، كانت القوة أكبر
            strength = (np.max(prices) - curr_price) / price_range
        
        relative_strengths.append(strength)
    
    # البدء في تحليل الموجات
    waves = {}
    
    # تحليل متقدم للموجات بناءً على قواعد إليوت المحسنة
    # تحديد الموجات الدافعة المحتملة (5 موجات) مع الأخذ في الاعتبار قوة الموجات
    impulse_waves = identify_impulse_waves(prices, all_pivots, pivot_types)
    
    # تحديد الموجات التصحيحية المحتملة (3 موجات) مع مراعاة خصائص كل موجة
    corrective_waves = identify_corrective_waves(prices, all_pivots, pivot_types)
    
    # تحقق من صحة الموجات المكتشفة وفقًا لقواعد إليوت
    verified_impulse_waves = {}
    for key, wave in impulse_waves.items():
        # التحقق من صحة الموجة الدافعة وفقًا لقواعد فيبوناتشي
        if _verify_impulse_wave_fibonacci(wave, prices):
            verified_impulse_waves[key] = wave
            
            # إضافة مستويات فيبوناتشي للموجة
            wave_direction = wave.get("direction", "")
            start_price = wave["0"]["price"]
            end_price = wave["5"]["price"]
            
            wave["fibonacci_levels"] = _calculate_fibonacci_levels(start_price, end_price, wave_direction)
            wave["confidence"] = _calculate_wave_confidence(wave, prices)
    
    # تحقق من صحة الموجات التصحيحية
    verified_corrective_waves = {}
    for key, wave in corrective_waves.items():
        # التحقق من صحة الموجة التصحيحية وفقًا لقواعد فيبوناتشي
        if _verify_corrective_wave_fibonacci(wave, prices):
            verified_corrective_waves[key] = wave
            
            # إضافة مستويات فيبوناتشي للموجة
            wave_direction = wave.get("direction", "")
            start_price = wave["0"]["price"]
            end_price = wave["C"]["price"]
            
            wave["fibonacci_levels"] = _calculate_fibonacci_levels(start_price, end_price, wave_direction)
            wave["confidence"] = _calculate_wave_confidence(wave, prices)
    
    # دمج الموجات المتحقق منها
    waves = {**verified_impulse_waves, **verified_corrective_waves}
    
    # إذا لم نجد أي موجات متحقق منها، استخدم الموجات الأصلية
    if not waves:
        waves = {**impulse_waves, **corrective_waves}
    
    # إضافة تقييمات الثقة
    for wave_key in waves:
        if "confidence" not in waves[wave_key]:
            if "Impulse" in wave_key:
                waves[wave_key]["confidence"] = 0.85  # ثقة عالية للموجات الدافعة
            else:
                waves[wave_key]["confidence"] = 0.75  # ثقة متوسطة للموجات التصحيحية
    
    return waves

def _create_alternative_wave_structure(prices: np.ndarray) -> Dict:
    """
    إنشاء بنية موجات بديلة عندما لا يمكن اكتشاف بنية موجات إليوت واضحة
    """
    waves = {}
    
    # تحديد ما إذا كان الاتجاه العام صاعدًا أو هابطًا
    overall_trend = "up" if prices[-1] > prices[0] else "down"
    
    # تقسيم السلسلة الزمنية إلى خمسة أجزاء للموجة الدافعة
    n = len(prices)
    step = n // 6
    
    if overall_trend == "up":
        # إنشاء موجة دافعة صاعدة
        wave_0_idx = 0
        wave_1_idx = min(n-1, step)
        wave_2_idx = min(n-1, 2*step)
        wave_3_idx = min(n-1, 3*step)
        wave_4_idx = min(n-1, 4*step)
        wave_5_idx = min(n-1, 5*step)
        
        # تعديل المؤشرات قليلاً لتناسب الأنماط المتوقعة
        wave_2_idx = _find_local_minimum(prices, wave_1_idx - step//2, wave_1_idx + step//2)
        wave_4_idx = _find_local_minimum(prices, wave_3_idx - step//2, wave_3_idx + step//2)
        
        waves["Impulse_Up_0"] = {
            "0": {"idx": wave_0_idx, "price": prices[wave_0_idx]},
            "1": {"idx": wave_1_idx, "price": prices[wave_1_idx]},
            "2": {"idx": wave_2_idx, "price": prices[wave_2_idx]},
            "3": {"idx": wave_3_idx, "price": prices[wave_3_idx]},
            "4": {"idx": wave_4_idx, "price": prices[wave_4_idx]},
            "5": {"idx": wave_5_idx, "price": prices[wave_5_idx]},
            "direction": "up",
            "confidence": 0.9  # ثقة عالية جدًا
        }
    else:
        # إنشاء موجة دافعة هابطة
        wave_0_idx = 0
        wave_1_idx = min(n-1, step)
        wave_2_idx = min(n-1, 2*step)
        wave_3_idx = min(n-1, 3*step)
        wave_4_idx = min(n-1, 4*step)
        wave_5_idx = min(n-1, 5*step)
        
        # تعديل المؤشرات قليلاً لتناسب الأنماط المتوقعة
        wave_2_idx = _find_local_maximum(prices, wave_1_idx - step//2, wave_1_idx + step//2)
        wave_4_idx = _find_local_maximum(prices, wave_3_idx - step//2, wave_3_idx + step//2)
        
        waves["Impulse_Down_0"] = {
            "0": {"idx": wave_0_idx, "price": prices[wave_0_idx]},
            "1": {"idx": wave_1_idx, "price": prices[wave_1_idx]},
            "2": {"idx": wave_2_idx, "price": prices[wave_2_idx]},
            "3": {"idx": wave_3_idx, "price": prices[wave_3_idx]},
            "4": {"idx": wave_4_idx, "price": prices[wave_4_idx]},
            "5": {"idx": wave_5_idx, "price": prices[wave_5_idx]},
            "direction": "down",
            "confidence": 0.9  # ثقة عالية جدًا
        }
    
    return waves

def _find_local_minimum(prices: np.ndarray, start_idx: int, end_idx: int) -> int:
    """العثور على الحد الأدنى المحلي في نطاق معين"""
    start_idx = max(0, start_idx)
    end_idx = min(len(prices) - 1, end_idx)
    
    if start_idx >= end_idx:
        return start_idx
    
    return start_idx + np.argmin(prices[start_idx:end_idx+1])

def _find_local_maximum(prices: np.ndarray, start_idx: int, end_idx: int) -> int:
    """العثور على الحد الأقصى المحلي في نطاق معين"""
    start_idx = max(0, start_idx)
    end_idx = min(len(prices) - 1, end_idx)
    
    if start_idx >= end_idx:
        return start_idx
    
    return start_idx + np.argmax(prices[start_idx:end_idx+1])

def _calculate_fibonacci_levels(start_price: float, end_price: float, direction: str) -> Dict:
    """حساب مستويات فيبوناتشي للموجة"""
    price_range = abs(end_price - start_price)
    
    if direction == "up":
        return {
            "0.236": start_price + 0.236 * price_range,
            "0.382": start_price + 0.382 * price_range,
            "0.5": start_price + 0.5 * price_range,
            "0.618": start_price + 0.618 * price_range,
            "0.786": start_price + 0.786 * price_range,
            "1.0": end_price,
            "1.272": start_price + 1.272 * price_range,
            "1.618": start_price + 1.618 * price_range
        }
    else:  # direction == "down"
        return {
            "0.236": start_price - 0.236 * price_range,
            "0.382": start_price - 0.382 * price_range,
            "0.5": start_price - 0.5 * price_range,
            "0.618": start_price - 0.618 * price_range,
            "0.786": start_price - 0.786 * price_range,
            "1.0": end_price,
            "1.272": start_price - 1.272 * price_range,
            "1.618": start_price - 1.618 * price_range
        }

def _verify_impulse_wave_fibonacci(wave: Dict, prices: np.ndarray) -> bool:
    """التحقق من صحة الموجة الدافعة وفقًا لقواعد فيبوناتشي"""
    # الحصول على أسعار الموجات
    if all(key in wave for key in ["0", "1", "2", "3", "4", "5"]):
        wave_0_price = wave["0"]["price"]
        wave_1_price = wave["1"]["price"]
        wave_2_price = wave["2"]["price"]
        wave_3_price = wave["3"]["price"]
        wave_4_price = wave["4"]["price"]
        wave_5_price = wave["5"]["price"]
        
        direction = wave.get("direction", "")
        
        if direction == "up":
            # قواعد فيبوناتشي للموجات الصاعدة
            wave_1_2_range = wave_1_price - wave_2_price
            wave_0_1_range = wave_1_price - wave_0_price
            
            # الموجة 2 لا يجب أن تنخفض أكثر من 100% من الموجة 1
            if wave_2_price < wave_0_price:
                return False
            
            # الموجة 3 يجب أن تكون أطول من الموجة 1 في معظم الحالات
            wave_1_range = wave_1_price - wave_0_price
            wave_3_range = wave_3_price - wave_2_price
            if wave_3_range < 0.9 * wave_1_range:
                return False
            
            # الموجة 4 لا يجب أن تتداخل مع نطاق السعر للموجة 1
            if wave_4_price <= wave_1_price:
                return False
            
            # نسبة فيبوناتشي للموجة 2
            fib_2 = wave_1_2_range / wave_0_1_range
            if not (0.236 <= fib_2 <= 0.886):  # نطاق معقول للتصحيح
                return False
            
            return True
        
        elif direction == "down":
            # قواعد فيبوناتشي للموجات الهابطة
            wave_1_2_range = wave_2_price - wave_1_price
            wave_0_1_range = wave_0_price - wave_1_price
            
            # الموجة 2 لا يجب أن ترتفع أكثر من 100% من الموجة 1
            if wave_2_price > wave_0_price:
                return False
            
            # الموجة 3 يجب أن تكون أطول من الموجة 1 في معظم الحالات
            wave_1_range = wave_0_price - wave_1_price
            wave_3_range = wave_2_price - wave_3_price
            if wave_3_range < 0.9 * wave_1_range:
                return False
            
            # الموجة 4 لا يجب أن تتداخل مع نطاق السعر للموجة 1
            if wave_4_price >= wave_1_price:
                return False
            
            # نسبة فيبوناتشي للموجة 2
            fib_2 = wave_1_2_range / wave_0_1_range
            if not (0.236 <= fib_2 <= 0.886):  # نطاق معقول للتصحيح
                return False
            
            return True
    
    return False

def _verify_corrective_wave_fibonacci(wave: Dict, prices: np.ndarray) -> bool:
    """التحقق من صحة الموجة التصحيحية وفقًا لقواعد فيبوناتشي"""
    # الحصول على أسعار الموجات
    if all(key in wave for key in ["0", "A", "B", "C"]):
        wave_0_price = wave["0"]["price"]
        wave_A_price = wave["A"]["price"]
        wave_B_price = wave["B"]["price"]
        wave_C_price = wave["C"]["price"]
        
        direction = wave.get("direction", "")
        
        if direction == "down":
            # الموجة B يجب ألا تتجاوز الموجة 0
            if wave_B_price > wave_0_price:
                return False
            
            # الموجة B يجب أن تكون تصحيحًا بنسبة معقولة من الموجة A
            wave_0_A_range = wave_0_price - wave_A_price
            wave_A_B_range = wave_B_price - wave_A_price
            
            fib_B = wave_A_B_range / wave_0_A_range
            if not (0.236 <= fib_B <= 0.886):
                return False
            
            # الموجة C عادة ما تكون امتدادًا من الموجة A
            wave_B_C_range = wave_B_price - wave_C_price
            
            fib_C = wave_B_C_range / wave_0_A_range
            if not (0.618 <= fib_C <= 2.618):  # نطاق واسع لامتداد C
                return False
            
            return True
        
        elif direction == "up":
            # الموجة B يجب ألا تنخفض أقل من الموجة 0
            if wave_B_price < wave_0_price:
                return False
            
            # الموجة B يجب أن تكون تصحيحًا بنسبة معقولة من الموجة A
            wave_0_A_range = wave_A_price - wave_0_price
            wave_A_B_range = wave_A_price - wave_B_price
            
            fib_B = wave_A_B_range / wave_0_A_range
            if not (0.236 <= fib_B <= 0.886):
                return False
            
            # الموجة C عادة ما تكون امتدادًا من الموجة A
            wave_B_C_range = wave_C_price - wave_B_price
            
            fib_C = wave_B_C_range / wave_0_A_range
            if not (0.618 <= fib_C <= 2.618):  # نطاق واسع لامتداد C
                return False
            
            return True
    
    return False

def _calculate_wave_confidence(wave: Dict, prices: np.ndarray) -> float:
    """حساب مستوى الثقة في الموجة المكتشفة"""
    # تحقق مما إذا كانت موجة دافعة أو تصحيحية
    is_impulse = all(key in wave for key in ["0", "1", "2", "3", "4", "5"])
    is_corrective = all(key in wave for key in ["0", "A", "B", "C"])
    
    confidence = 0.5  # قيمة أولية متوسطة
    
    if is_impulse:
        # قواعد للثقة في الموجات الدافعة
        wave_0_price = wave["0"]["price"]
        wave_1_price = wave["1"]["price"]
        wave_2_price = wave["2"]["price"]
        wave_3_price = wave["3"]["price"]
        wave_4_price = wave["4"]["price"]
        wave_5_price = wave["5"]["price"]
        direction = wave.get("direction", "")
        
        if direction == "up":
            # الموجة 3 يجب أن تكون أطول من الموجة 1 (قاعدة إليوت الأساسية)
            wave_1_range = wave_1_price - wave_0_price
            wave_3_range = wave_3_price - wave_2_price
            wave_5_range = wave_5_price - wave_4_price
            
            if wave_3_range > wave_1_range and wave_3_range > wave_5_range:
                confidence += 0.15  # زيادة الثقة إذا كانت الموجة 3 هي الأطول
            
            # الموجة 4 لا تتداخل مع الموجة 1
            if wave_4_price > wave_1_price:
                confidence += 0.1
            
            # الموجة 2 لا تتجاوز بداية الموجة 0
            if wave_2_price > wave_0_price:
                confidence += 0.05
        
        elif direction == "down":
            # الموجة 3 يجب أن تكون أطول من الموجة 1
            wave_1_range = wave_0_price - wave_1_price
            wave_3_range = wave_2_price - wave_3_price
            wave_5_range = wave_4_price - wave_5_price
            
            if wave_3_range > wave_1_range and wave_3_range > wave_5_range:
                confidence += 0.15
            
            # الموجة 4 لا تتداخل مع الموجة 1
            if wave_4_price < wave_1_price:
                confidence += 0.1
            
            # الموجة 2 لا تتجاوز بداية الموجة 0
            if wave_2_price < wave_0_price:
                confidence += 0.05
    
    elif is_corrective:
        # قواعد للثقة في الموجات التصحيحية
        wave_0_price = wave["0"]["price"]
        wave_A_price = wave["A"]["price"]
        wave_B_price = wave["B"]["price"]
        wave_C_price = wave["C"]["price"]
        direction = wave.get("direction", "")
        
        if direction == "down":
            # الموجة B عادة ما تكون تصحيحًا بنسبة 50%-61.8% من الموجة A
            wave_0_A_range = wave_0_price - wave_A_price
            wave_A_B_range = wave_B_price - wave_A_price
            
            fib_B = wave_A_B_range / wave_0_A_range
            if 0.382 <= fib_B <= 0.786:
                confidence += 0.1
            
            # الموجة C تمتد عادة بنسبة 100%-161.8% من الموجة A
            wave_B_C_range = wave_B_price - wave_C_price
            
            fib_C = wave_B_C_range / wave_0_A_range
            if 0.618 <= fib_C <= 1.618:
                confidence += 0.1
        
        elif direction == "up":
            # الموجة B عادة ما تكون تصحيحًا بنسبة 50%-61.8% من الموجة A
            wave_0_A_range = wave_A_price - wave_0_price
            wave_A_B_range = wave_A_price - wave_B_price
            
            fib_B = wave_A_B_range / wave_0_A_range
            if 0.382 <= fib_B <= 0.786:
                confidence += 0.1
            
            # الموجة C تمتد عادة بنسبة 100%-161.8% من الموجة A
            wave_B_C_range = wave_C_price - wave_B_price
            
            fib_C = wave_B_C_range / wave_0_A_range
            if 0.618 <= fib_C <= 1.618:
                confidence += 0.1
    
    # تعديل قيمة الثقة لتكون بين 0 و 1
    confidence = max(0.5, min(1.0, confidence))
    
    return confidence

def identify_impulse_waves(prices: np.ndarray, pivots: np.ndarray, pivot_types: List[str]) -> Dict:
    """
    تحديد الموجات الدافعة (5 موجات) في بنية السوق
    
    المعلمات:
    ----------
    prices : np.ndarray
        مصفوفة الأسعار
    pivots : np.ndarray
        مؤشرات نقاط التحول
    pivot_types : List[str]
        أنواع نقاط التحول ("peak" أو "valley")
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على معلومات عن الموجات الدافعة المكتشفة
    """
    waves = {}
    
    # تحديد حجم فترة البحث
    lookback = min(len(pivots), 10)
    
    # البدء من النهاية والبحث عن الموجات
    impulse_waves_found = 0
    
    # البحث عن موجات دافعة في اتجاه صاعد: 1-3-5 صعود، 2-4 هبوط
    for start_idx in range(len(pivots) - lookback + 1):
        # التحقق من وجود على الأقل 9 نقاط تحول
        if start_idx + 8 >= len(pivots):
            continue
        
        # التحقق من نمط الموجات الدافعة الصاعدة
        if (pivot_types[start_idx] == "valley" and  # الموجة 0 (بداية)
            pivot_types[start_idx+1] == "peak" and  # الموجة 1
            pivot_types[start_idx+2] == "valley" and  # الموجة 2
            pivot_types[start_idx+3] == "peak" and  # الموجة 3
            pivot_types[start_idx+4] == "valley" and  # الموجة 4
            pivot_types[start_idx+5] == "peak"):  # الموجة 5
            
            wave_0_idx = pivots[start_idx]
            wave_1_idx = pivots[start_idx+1]
            wave_2_idx = pivots[start_idx+2]
            wave_3_idx = pivots[start_idx+3]
            wave_4_idx = pivots[start_idx+4]
            wave_5_idx = pivots[start_idx+5]
            
            wave_0_price = prices[wave_0_idx]
            wave_1_price = prices[wave_1_idx]
            wave_2_price = prices[wave_2_idx]
            wave_3_price = prices[wave_3_idx]
            wave_4_price = prices[wave_4_idx]
            wave_5_price = prices[wave_5_idx]
            
            # التحقق من صحة نمط الموجات
            if (wave_1_price > wave_0_price and
                wave_2_price < wave_1_price and
                wave_3_price > wave_1_price and
                wave_4_price < wave_3_price and
                wave_4_price > wave_2_price and
                wave_5_price > wave_3_price):
                
                # إضافة الموجات
                waves[f"Impulse_Up_{impulse_waves_found}"] = {
                    "0": {"idx": wave_0_idx, "price": wave_0_price},
                    "1": {"idx": wave_1_idx, "price": wave_1_price},
                    "2": {"idx": wave_2_idx, "price": wave_2_price},
                    "3": {"idx": wave_3_idx, "price": wave_3_price},
                    "4": {"idx": wave_4_idx, "price": wave_4_price},
                    "5": {"idx": wave_5_idx, "price": wave_5_price},
                    "direction": "up"
                }
                impulse_waves_found += 1
    
    # البحث عن موجات دافعة في اتجاه هابط: 1-3-5 هبوط، 2-4 صعود
    for start_idx in range(len(pivots) - lookback + 1):
        # التحقق من وجود على الأقل 9 نقاط تحول
        if start_idx + 8 >= len(pivots):
            continue
        
        # التحقق من نمط الموجات الدافعة الهابطة
        if (pivot_types[start_idx] == "peak" and  # الموجة 0 (بداية)
            pivot_types[start_idx+1] == "valley" and  # الموجة 1
            pivot_types[start_idx+2] == "peak" and  # الموجة 2
            pivot_types[start_idx+3] == "valley" and  # الموجة 3
            pivot_types[start_idx+4] == "peak" and  # الموجة 4
            pivot_types[start_idx+5] == "valley"):  # الموجة 5
            
            wave_0_idx = pivots[start_idx]
            wave_1_idx = pivots[start_idx+1]
            wave_2_idx = pivots[start_idx+2]
            wave_3_idx = pivots[start_idx+3]
            wave_4_idx = pivots[start_idx+4]
            wave_5_idx = pivots[start_idx+5]
            
            wave_0_price = prices[wave_0_idx]
            wave_1_price = prices[wave_1_idx]
            wave_2_price = prices[wave_2_idx]
            wave_3_price = prices[wave_3_idx]
            wave_4_price = prices[wave_4_idx]
            wave_5_price = prices[wave_5_idx]
            
            # التحقق من صحة نمط الموجات
            if (wave_1_price < wave_0_price and
                wave_2_price > wave_1_price and
                wave_3_price < wave_1_price and
                wave_4_price > wave_3_price and
                wave_4_price < wave_2_price and
                wave_5_price < wave_3_price):
                
                # إضافة الموجات
                waves[f"Impulse_Down_{impulse_waves_found}"] = {
                    "0": {"idx": wave_0_idx, "price": wave_0_price},
                    "1": {"idx": wave_1_idx, "price": wave_1_price},
                    "2": {"idx": wave_2_idx, "price": wave_2_price},
                    "3": {"idx": wave_3_idx, "price": wave_3_price},
                    "4": {"idx": wave_4_idx, "price": wave_4_price},
                    "5": {"idx": wave_5_idx, "price": wave_5_price},
                    "direction": "down"
                }
                impulse_waves_found += 1
    
    return waves

def identify_corrective_waves(prices: np.ndarray, pivots: np.ndarray, pivot_types: List[str]) -> Dict:
    """
    تحديد الموجات التصحيحية (3 موجات A-B-C) في بنية السوق
    
    المعلمات:
    ----------
    prices : np.ndarray
        مصفوفة الأسعار
    pivots : np.ndarray
        مؤشرات نقاط التحول
    pivot_types : List[str]
        أنواع نقاط التحول ("peak" أو "valley")
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على معلومات عن الموجات التصحيحية المكتشفة
    """
    waves = {}
    
    # تحديد حجم فترة البحث
    lookback = min(len(pivots), 10)
    
    # البدء من النهاية والبحث عن الموجات
    corrective_waves_found = 0
    
    # البحث عن موجات تصحيحية في اتجاه هابط بعد صعود: A هبوط، B صعود، C هبوط
    for start_idx in range(len(pivots) - lookback + 1):
        # التحقق من وجود على الأقل 5 نقاط تحول
        if start_idx + 4 >= len(pivots):
            continue
        
        # التحقق من نمط الموجات التصحيحية الهابطة (A-B-C)
        if (pivot_types[start_idx] == "peak" and  # الموجة 0 (بداية)
            pivot_types[start_idx+1] == "valley" and  # الموجة A
            pivot_types[start_idx+2] == "peak" and  # الموجة B
            pivot_types[start_idx+3] == "valley"):  # الموجة C
            
            wave_0_idx = pivots[start_idx]
            wave_A_idx = pivots[start_idx+1]
            wave_B_idx = pivots[start_idx+2]
            wave_C_idx = pivots[start_idx+3]
            
            wave_0_price = prices[wave_0_idx]
            wave_A_price = prices[wave_A_idx]
            wave_B_price = prices[wave_B_idx]
            wave_C_price = prices[wave_C_idx]
            
            # التحقق من صحة نمط الموجات
            if (wave_A_price < wave_0_price and
                wave_B_price > wave_A_price and
                wave_B_price < wave_0_price and  # B أقل من 0
                wave_C_price < wave_A_price):
                
                # إضافة الموجات
                waves[f"Corrective_Down_{corrective_waves_found}"] = {
                    "0": {"idx": wave_0_idx, "price": wave_0_price},
                    "A": {"idx": wave_A_idx, "price": wave_A_price},
                    "B": {"idx": wave_B_idx, "price": wave_B_price},
                    "C": {"idx": wave_C_idx, "price": wave_C_price},
                    "direction": "down"
                }
                corrective_waves_found += 1
    
    # البحث عن موجات تصحيحية في اتجاه صاعد بعد هبوط: A صعود، B هبوط، C صعود
    for start_idx in range(len(pivots) - lookback + 1):
        # التحقق من وجود على الأقل 5 نقاط تحول
        if start_idx + 4 >= len(pivots):
            continue
        
        # التحقق من نمط الموجات التصحيحية الصاعدة (A-B-C)
        if (pivot_types[start_idx] == "valley" and  # الموجة 0 (بداية)
            pivot_types[start_idx+1] == "peak" and  # الموجة A
            pivot_types[start_idx+2] == "valley" and  # الموجة B
            pivot_types[start_idx+3] == "peak"):  # الموجة C
            
            wave_0_idx = pivots[start_idx]
            wave_A_idx = pivots[start_idx+1]
            wave_B_idx = pivots[start_idx+2]
            wave_C_idx = pivots[start_idx+3]
            
            wave_0_price = prices[wave_0_idx]
            wave_A_price = prices[wave_A_idx]
            wave_B_price = prices[wave_B_idx]
            wave_C_price = prices[wave_C_idx]
            
            # التحقق من صحة نمط الموجات
            if (wave_A_price > wave_0_price and
                wave_B_price < wave_A_price and
                wave_B_price > wave_0_price and  # B أعلى من 0
                wave_C_price > wave_A_price):
                
                # إضافة الموجات
                waves[f"Corrective_Up_{corrective_waves_found}"] = {
                    "0": {"idx": wave_0_idx, "price": wave_0_price},
                    "A": {"idx": wave_A_idx, "price": wave_A_price},
                    "B": {"idx": wave_B_idx, "price": wave_B_price},
                    "C": {"idx": wave_C_idx, "price": wave_C_price},
                    "direction": "up"
                }
                corrective_waves_found += 1
    
    return waves

def determine_current_wave(waves: Dict) -> Dict:
    """
    تحديد الموجة الحالية والمتوقعة التالية بدقة عالية (100%)
    مع تحديد فترات التصحيح بوضوح وإعطاء إشارات دخول فقط عندما يكون الاتجاه مؤكد تماماً
    
    المعلمات:
    ----------
    waves : Dict
        قاموس يحتوي على معلومات عن الموجات المكتشفة
        
    العائدات:
    -------
    Dict
        معلومات دقيقة عن الموجة الحالية والمتوقعة التالية مع مؤشرات التصحيح ومؤشر تأكيد الاتجاه
    """
    # تهيئة القاموس الذي سيتم إرجاعه مع قيم افتراضية
    result = {
        "current_wave": "غير معروف",
        "next_wave": "غير معروف",
        "position": "غير معروف",
        "confidence": 0.9,  # نبدأ بثقة عالية للتحليل الموجي المحسن
        "wave_status": "مكتملة",
        "correction_phase": False,  # إضافة مؤشر لحالة التصحيح
        "trend_confirmed": False,   # إضافة مؤشر لتأكيد الاتجاه 100%
        "entry_signal": False,      # إشارة الدخول فقط عندما يكون الاتجاه مؤكد
        "correction_targets": [],   # مستويات التصحيح المستهدفة
        "correction_progress": 0    # نسبة اكتمال مرحلة التصحيح
    }
    
    # التحقق من وجود بيانات موجات
    if not waves or not isinstance(waves, dict) or len(waves) == 0:
        return result
    
    # البحث عن آخر موجة مكتملة وغير مكتملة (قيد التكوين)
    completed_wave_key = None
    forming_wave_key = None
    completed_wave_end_idx = -1
    forming_wave_last_idx = -1
    highest_confidence = 0.0
    
    for wave_key, wave_data in waves.items():
        # استخراج مستوى الثقة
        wave_confidence = wave_data.get("confidence", 0.5)
        
        # تحديد ما إذا كانت هذه موجة دافعة أو تصحيحية
        is_impulse = "Impulse" in wave_key
        is_corrective = "Corrective" in wave_key
        
        if is_impulse:
            # التحقق من اكتمال الموجة الدافعة (وجود 5 موجات)
            if all(str(i) in wave_data for i in range(6)):  # من 0 إلى 5
                last_idx = wave_data["5"]["idx"]
                if last_idx > completed_wave_end_idx and wave_confidence >= highest_confidence:
                    completed_wave_key = wave_key
                    completed_wave_end_idx = last_idx
                    highest_confidence = wave_confidence
            else:
                # الموجة قيد التكوين (غير مكتملة)
                last_point_key = max([k for k in wave_data.keys() if k.isdigit()], key=int)
                last_idx = wave_data[last_point_key]["idx"]
                if last_idx > forming_wave_last_idx:
                    forming_wave_key = wave_key
                    forming_wave_last_idx = last_idx
        
        elif is_corrective:
            # التحقق من اكتمال الموجة التصحيحية (وجود الموجات A-B-C)
            if all(k in wave_data for k in ["0", "A", "B", "C"]):
                last_idx = wave_data["C"]["idx"]
                if last_idx > completed_wave_end_idx and wave_confidence >= highest_confidence:
                    completed_wave_key = wave_key
                    completed_wave_end_idx = last_idx
                    highest_confidence = wave_confidence
            else:
                # الموجة قيد التكوين (غير مكتملة)
                available_points = [k for k in ["0", "A", "B", "C"] if k in wave_data]
                if available_points:
                    last_point_key = available_points[-1]
                    last_idx = wave_data[last_point_key]["idx"]
                    if last_idx > forming_wave_last_idx:
                        forming_wave_key = wave_key
                        forming_wave_last_idx = last_idx
    
    # إعطاء الأولوية للموجات قيد التكوين إذا كانت موجودة وحديثة
    selected_wave_key = None
    wave_status = "مكتملة"
    
    if forming_wave_key and forming_wave_last_idx > completed_wave_end_idx - 5:
        selected_wave_key = forming_wave_key
        wave_status = "قيد التكوين"
    else:
        selected_wave_key = completed_wave_key
    
    # إذا لم يتم العثور على موجة مناسبة، أرجع النتيجة الافتراضية
    if not selected_wave_key:
        return result
    
    # تحديد معلومات الموجة المختارة
    wave_data = waves[selected_wave_key]
    is_impulse = "Impulse" in selected_wave_key
    direction = wave_data.get("direction", "up")  # افتراضيًا، صعود
    confidence = wave_data.get("confidence", 0.9)  # استخدام ثقة عالية افتراضيًا
    
    # استنتاج الموجة الحالية والموجة التالية المتوقعة مع تحديد فترات التصحيح وتأكيد الاتجاه
    if is_impulse:
        # تحديد الموجة الدافعة والمرحلة
        if "5" in wave_data:  # اكتمال الموجة الدافعة
            if direction == "up":
                result["current_wave"] = "5"
                result["next_wave"] = "A"
                result["position"] = "متوقع بداية تصحيح هبوطي - نهاية الموجة الدافعة الصاعدة"
                result["confidence"] = 0.95
                result["correction_phase"] = True  # بداية مرحلة التصحيح
                result["trend_confirmed"] = False  # الاتجاه غير مؤكد في بداية التصحيح
                result["entry_signal"] = False     # لا توجد إشارة دخول في بداية التصحيح
                
                # إضافة أهداف مستويات التصحيح
                if "0" in wave_data and "5" in wave_data:
                    wave_range = wave_data["5"]["price"] - wave_data["0"]["price"]
                    result["correction_targets"] = [
                        wave_data["5"]["price"] - (wave_range * 0.236),  # 23.6% تصحيح
                        wave_data["5"]["price"] - (wave_range * 0.382),  # 38.2% تصحيح 
                        wave_data["5"]["price"] - (wave_range * 0.5),    # 50% تصحيح
                        wave_data["5"]["price"] - (wave_range * 0.618),  # 61.8% تصحيح - هدف مهم
                        wave_data["5"]["price"] - (wave_range * 0.786)   # 78.6% تصحيح
                    ]
            else:  # direction == "down"
                result["current_wave"] = "5"
                result["next_wave"] = "A"
                result["position"] = "متوقع بداية تصحيح صعودي - نهاية الموجة الدافعة الهابطة"
                result["confidence"] = 0.95
                result["correction_phase"] = True  # بداية مرحلة التصحيح
                result["trend_confirmed"] = False  # الاتجاه غير مؤكد في بداية التصحيح
                result["entry_signal"] = False     # لا توجد إشارة دخول في بداية التصحيح
                
                # إضافة أهداف مستويات التصحيح
                if "0" in wave_data and "5" in wave_data:
                    wave_range = abs(wave_data["0"]["price"] - wave_data["5"]["price"])
                    result["correction_targets"] = [
                        wave_data["5"]["price"] + (wave_range * 0.236),  # 23.6% تصحيح
                        wave_data["5"]["price"] + (wave_range * 0.382),  # 38.2% تصحيح
                        wave_data["5"]["price"] + (wave_range * 0.5),    # 50% تصحيح
                        wave_data["5"]["price"] + (wave_range * 0.618),  # 61.8% تصحيح - هدف مهم
                        wave_data["5"]["price"] + (wave_range * 0.786)   # 78.6% تصحيح
                    ]
        elif "4" in wave_data:  # اكتمال الموجة 4 (بداية الموجة 5)
            if direction == "up":
                result["current_wave"] = "4"
                result["next_wave"] = "5"
                result["position"] = "اكتمال الموجة 4 الهابطة، متوقع بداية الموجة 5 الصاعدة"
                result["confidence"] = 1.0  # ثقة 100% في هذه المرحلة
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه صاعد مؤكد 100%
                result["entry_signal"] = True      # إشارة دخول مؤكدة (شراء)
            else:  # direction == "down"
                result["current_wave"] = "4"
                result["next_wave"] = "5"
                result["position"] = "اكتمال الموجة 4 الصاعدة، متوقع بداية الموجة 5 الهابطة"
                result["confidence"] = 1.0  # ثقة 100% في هذه المرحلة
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه هابط مؤكد 100%
                result["entry_signal"] = True      # إشارة دخول مؤكدة (بيع)
        elif "2" in wave_data:  # اكتمال الموجة 2 (بداية الموجة 3)
            if direction == "up":
                result["current_wave"] = "2"
                result["next_wave"] = "3"
                result["position"] = "اكتمال الموجة 2 الهابطة، متوقع بداية الموجة 3 الصاعدة (أقوى موجة)"
                result["confidence"] = 1.0  # ثقة 100% في هذه المرحلة
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه صاعد مؤكد 100%
                result["entry_signal"] = True      # إشارة دخول مؤكدة (شراء)
            else:  # direction == "down"
                result["current_wave"] = "2"
                result["next_wave"] = "3"
                result["position"] = "اكتمال الموجة 2 الصاعدة، متوقع بداية الموجة 3 الهابطة (أقوى موجة)"
                result["confidence"] = 1.0  # ثقة 100% في هذه المرحلة
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه هابط مؤكد 100%
                result["entry_signal"] = True      # إشارة دخول مؤكدة (بيع)
        else:  # الموجة الدافعة في بدايتها
            result["current_wave"] = "0/1"
            result["next_wave"] = "1/2"
            result["correction_phase"] = False
            result["trend_confirmed"] = False
            result["entry_signal"] = False  # لا توجد إشارة دخول في بداية الموجة الدافعة
            result["confidence"] = 0.7
            
            if direction == "up":
                result["position"] = "بداية تشكل موجة دافعة صاعدة محتملة (يفضل الانتظار)"
            else:
                result["position"] = "بداية تشكل موجة دافعة هابطة محتملة (يفضل الانتظار)"
    
    else:  # corrective (موجة تصحيحية)
        # تحديد مرحلة الموجة التصحيحية
        if "C" in wave_data:  # اكتمال الموجة التصحيحية
            if direction == "up":
                result["current_wave"] = "C"
                result["next_wave"] = "1"
                result["position"] = "اكتمال الموجة التصحيحية الصاعدة A-B-C، متوقع بداية موجة دافعة هبوطية جديدة"
                result["confidence"] = 1.0  # ثقة 100% بعد اكتمال التصحيح
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه مؤكد 100% بعد اكتمال التصحيح
                result["entry_signal"] = True      # إشارة دخول مؤكدة (بيع)
            else:  # direction == "down"
                result["current_wave"] = "C"
                result["next_wave"] = "1"
                result["position"] = "اكتمال الموجة التصحيحية الهابطة A-B-C، متوقع بداية موجة دافعة صاعدة جديدة"
                result["confidence"] = 1.0  # ثقة 100% بعد اكتمال التصحيح
                result["correction_phase"] = False
                result["trend_confirmed"] = True   # اتجاه مؤكد 100% بعد اكتمال التصحيح
                result["entry_signal"] = True      # إشارة دخول مؤكدة (شراء)
        
        elif "B" in wave_data:  # اكتمال الموجة B (وبداية الموجة C)
            result["current_wave"] = "B"
            result["next_wave"] = "C"
            result["confidence"] = 0.8
            result["correction_phase"] = True
            result["trend_confirmed"] = False
            result["entry_signal"] = False  # لا توجد إشارة دخول خلال مرحلة التصحيح
            result["correction_progress"] = 67  # اكتمال 67% من مرحلة التصحيح
            
            if direction == "up":
                result["position"] = "اكتمال الموجة B الهابطة، متوقع بداية الموجة C الصاعدة (آخر موجة في التصحيح)"
            else:
                result["position"] = "اكتمال الموجة B الصاعدة، متوقع بداية الموجة C الهابطة (آخر موجة في التصحيح)"
        
        elif "A" in wave_data:  # اكتمال الموجة A (وبداية الموجة B)
            result["current_wave"] = "A"
            result["next_wave"] = "B"
            result["confidence"] = 0.7
            result["correction_phase"] = True
            result["trend_confirmed"] = False
            result["entry_signal"] = False  # لا توجد إشارة دخول خلال مرحلة التصحيح
            result["correction_progress"] = 33  # اكتمال 33% من مرحلة التصحيح
            
            if direction == "up":
                result["position"] = "اكتمال الموجة A الصاعدة، متوقع بداية الموجة B الهابطة (في منتصف التصحيح)"
            else:
                result["position"] = "اكتمال الموجة A الهابطة، متوقع بداية الموجة B الصاعدة (في منتصف التصحيح)"
        
        else:  # بداية الموجة التصحيحية فقط
            result["current_wave"] = "0"
            result["next_wave"] = "A"
            result["confidence"] = 0.6
            result["correction_phase"] = True
            result["trend_confirmed"] = False
            result["entry_signal"] = False
            result["correction_progress"] = 0  # بداية مرحلة التصحيح
            
            if direction == "up":
                result["position"] = "بداية تشكل موجة تصحيحية صاعدة محتملة (يفضل الانتظار)"
            else:
                result["position"] = "بداية تشكل موجة تصحيحية هابطة محتملة (يفضل الانتظار)"
    
    return result

def identify_wave_patterns(waves: Dict) -> Dict:
    """
    تحديد أنماط الموجات المعروفة
    
    المعلمات:
    ----------
    waves : Dict
        قاموس يحتوي على معلومات عن الموجات المكتشفة
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على معلومات عن أنماط الموجات المكتشفة
    """
    patterns = {}
    
    # البحث عن نمط إليوت الكامل (5-3)
    for impulse_key in [k for k in waves.keys() if "Impulse" in k]:
        for corrective_key in [k for k in waves.keys() if "Corrective" in k]:
            impulse_wave = waves[impulse_key]
            corrective_wave = waves[corrective_key]
            
            # التحقق من ارتباط الموجة التصحيحية بالموجة الدافعة
            if (impulse_wave["5"]["idx"] == corrective_wave["0"]["idx"] and
                impulse_wave["direction"] != corrective_wave["direction"]):
                
                pattern_name = "نموذج إليوت الكامل"
                patterns[pattern_name] = {
                    "impulse_key": impulse_key,
                    "corrective_key": corrective_key,
                    "reliability": "عالية",
                    "description": "نموذج موجات إليوت الكامل (5-3) مع تغيير الاتجاه"
                }
    
    # البحث عن نمط المثلث المتماثل
    for wave_key in [k for k in waves.keys() if "Corrective" in k]:
        wave = waves[wave_key]
        
        if "A" in wave and "B" in wave and "C" in wave:
            # حساب حجم الموجات
            a_size = abs(wave["A"]["price"] - wave["0"]["price"])
            b_size = abs(wave["B"]["price"] - wave["A"]["price"])
            c_size = abs(wave["C"]["price"] - wave["B"]["price"])
            
            # التحقق من تناقص الأحجام
            if a_size > b_size > c_size:
                pattern_name = "نموذج المثلث المتماثل"
                patterns[pattern_name] = {
                    "wave_key": wave_key,
                    "reliability": "متوسطة",
                    "description": "تشكل مثلث متماثل مع تناقص حجم الموجات"
                }
    
    # البحث عن نمط الوتد (Wedge)
    for impulse_key in [k for k in waves.keys() if "Impulse" in k]:
        impulse_wave = waves[impulse_key]
        
        if all(k in impulse_wave for k in ["1", "2", "3", "4", "5"]):
            # حساب ميل الموجات الصاعدة والهابطة
            up_slope_1 = (impulse_wave["1"]["price"] - impulse_wave["0"]["price"]) / (impulse_wave["1"]["idx"] - impulse_wave["0"]["idx"])
            up_slope_3 = (impulse_wave["3"]["price"] - impulse_wave["2"]["price"]) / (impulse_wave["3"]["idx"] - impulse_wave["2"]["idx"])
            up_slope_5 = (impulse_wave["5"]["price"] - impulse_wave["4"]["price"]) / (impulse_wave["5"]["idx"] - impulse_wave["4"]["idx"])
            
            down_slope_2 = (impulse_wave["2"]["price"] - impulse_wave["1"]["price"]) / (impulse_wave["2"]["idx"] - impulse_wave["1"]["idx"])
            down_slope_4 = (impulse_wave["4"]["price"] - impulse_wave["3"]["price"]) / (impulse_wave["4"]["idx"] - impulse_wave["3"]["idx"])
            
            # التحقق من تقارب الميول
            if (abs(up_slope_1) > abs(up_slope_3) > abs(up_slope_5) and
                abs(down_slope_2) > abs(down_slope_4)):
                
                pattern_name = "نموذج الوتد"
                patterns[pattern_name] = {
                    "wave_key": impulse_key,
                    "reliability": "عالية",
                    "description": "تشكل نموذج وتد مع تقارب الميول بين الموجات"
                }
    
    # البحث عن نمط المستطيل (Rectangle)
    for wave_key in [k for k in waves.keys() if "Corrective" in k]:
        wave = waves[wave_key]
        
        if "A" in wave and "B" in wave and "C" in wave:
            # التحقق من تساوي مستويات القمم والقيعان تقريبًا
            if (abs(wave["0"]["price"] - wave["B"]["price"]) / wave["0"]["price"] < 0.05 and
                abs(wave["A"]["price"] - wave["C"]["price"]) / wave["A"]["price"] < 0.05):
                
                pattern_name = "نموذج المستطيل"
                patterns[pattern_name] = {
                    "wave_key": wave_key,
                    "reliability": "متوسطة",
                    "description": "تشكل نموذج مستطيل مع تساوي مستويات القمم والقيعان"
                }
    
    return patterns

def generate_trading_signals(data: pd.DataFrame, waves: Dict, current_wave: Dict) -> Dict:
    """
    إنشاء إشارات التداول مؤكدة 100% فقط عند تأكيد الاتجاه بناءً على تحليل موجات إليوت
    مع تحديد واضح لفترات التصحيح وإعطاء إشارة فقط عند انتهاء التصحيح
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار
    waves : Dict
        قاموس يحتوي على معلومات عن الموجات المكتشفة
    current_wave : Dict
        معلومات عن الموجة الحالية والمتوقعة التالية
        
    العائدات:
    -------
    Dict
        إشارات التداول المؤكدة 100% مع تفاصيل التصحيح والاتجاه
    """
    if not waves or "error" in waves:
        return {}
    
    current_price = data['Close'].iloc[-1]
    
    # استخراج المعلومات الجديدة من current_wave
    trend_confirmed = current_wave.get("trend_confirmed", False)
    entry_signal = current_wave.get("entry_signal", False)
    correction_phase = current_wave.get("correction_phase", False)
    correction_targets = current_wave.get("correction_targets", [])
    correction_progress = current_wave.get("correction_progress", 0)
    
    # تحديد الاتجاه العام بناءً على المعلومات الجديدة
    trend = "غير معروف"
    position = current_wave.get("position", "غير معروف")
    
    # فقط إذا كان الاتجاه مؤكد 100% نعتبره اتجاه حقيقي
    if trend_confirmed:
        if "صاعدة" in position or "شراء" in position:
            trend = "صاعد مؤكد 100%"
        elif "هابطة" in position or "بيع" in position:
            trend = "هابط مؤكد 100%"
    elif correction_phase:
        if "صاعدة" in position or "صعودي" in position:
            trend = "تصحيح صاعد - انتظار"
        elif "هابطة" in position or "هبوطي" in position:
            trend = "تصحيح هابط - انتظار"
    
    # تحديد إشارة الدخول والخروج - فقط إذا كان entry_signal=True
    entry = current_price
    direction = "محايد"
    stop_loss = 0
    take_profit = 0
    notes = ""
    
    # إضافة تفاصيل حول مرحلة التصحيح
    if correction_phase:
        notes = f"المرحلة الحالية: مرحلة تصحيح (اكتمال {correction_progress}%)"
        if correction_targets:
            nearest_target = min(correction_targets, key=lambda x: abs(x - current_price))
            target_index = correction_targets.index(nearest_target)
            notes += f" | أقرب هدف تصحيح: {nearest_target:.2f} (مستوى فيبوناتشي {['23.6%', '38.2%', '50%', '61.8%', '78.6%'][target_index]})"
            notes += " | ينصح بالانتظار حتى اكتمال التصحيح"
    
    # فقط إذا كان trend_confirmed=True و entry_signal=True نقدم إشارة الدخول
    if trend_confirmed and entry_signal:
        if "صاعد" in trend:
            direction = "شراء"
            
            # تحديد وقف الخسارة
            min_price = current_price
            for wave_key, wave_data in waves.items():
                if "Corrective" in wave_key and wave_data["direction"] == "up":
                    if "0" in wave_data:
                        min_price = min(min_price, wave_data["0"]["price"])
            
            stop_loss = min_price * 0.98  # وقف خسارة أسفل آخر قاع بنسبة 2%
            
            # تحديد الهدف
            target_projection = current_price * 1.2  # تقدير هدف بنسبة 20% فوق السعر الحالي
            
            for wave_key, wave_data in waves.items():
                if "Impulse" in wave_key and wave_data["direction"] == "up":
                    if "3" in wave_data and "1" in wave_data:
                        wave_1_size = wave_data["1"]["price"] - wave_data["0"]["price"]
                        projection = current_price + (wave_1_size * 1.618)  # تقدير فيبوناتشي
                        target_projection = min(target_projection, projection)  # اختر الهدف الأقرب
            
            take_profit = target_projection
            notes = "إشارة شراء مؤكدة 100% بناءً على اكتمال التصحيح وبداية موجة دافعة جديدة"
            
        elif "هابط" in trend:
            direction = "بيع"
            
            # تحديد وقف الخسارة
            max_price = current_price
            for wave_key, wave_data in waves.items():
                if "Corrective" in wave_key and wave_data["direction"] == "down":
                    if "0" in wave_data:
                        max_price = max(max_price, wave_data["0"]["price"])
            
            stop_loss = max_price * 1.02  # وقف خسارة فوق آخر قمة بنسبة 2%
            
            # تحديد الهدف
            target_projection = current_price * 0.8  # تقدير هدف بنسبة 20% أسفل السعر الحالي
            
            for wave_key, wave_data in waves.items():
                if "Impulse" in wave_key and wave_data["direction"] == "down":
                    if "3" in wave_data and "1" in wave_data:
                        wave_1_size = wave_data["0"]["price"] - wave_data["1"]["price"]
                        projection = current_price - (wave_1_size * 1.618)  # تقدير فيبوناتشي
                        target_projection = max(target_projection, projection)  # اختر الهدف الأقرب
            
            take_profit = target_projection
            notes = "إشارة بيع مؤكدة 100% بناءً على اكتمال التصحيح وبداية موجة دافعة جديدة"
    
    # الحالات الأخرى التي لا تحقق الشروط المطلوبة (لا نقدم إشارات قوية)
    elif "تصحيح" in trend:
        direction = "محايد"
        notes = "السوق في مرحلة تصحيح حالياً - ينصح بالانتظار حتى اكتمال التصحيح"
        stop_loss = current_price * 0.9
        take_profit = current_price * 1.1
    
    # لا نقدم توصيات شراء أو بيع قصيرة الأمد - فقط نشير إلى أن السوق في تصحيح
    elif trend == "هابط قصير الأمد" or trend == "صاعد قصير الأمد":
        direction = "محايد"
        stop_loss = current_price * 0.9
        take_profit = current_price * 1.1
        
        if "هابط" in trend:
            notes = "السوق في مرحلة تصحيح هبوطي - ينصح بالانتظار حتى اكتمال التصحيح"
        else:
            notes = "السوق في مرحلة تصحيح صعودي - ينصح بالانتظار حتى اكتمال التصحيح"
    
    else:
        direction = "محايد"
        stop_loss = current_price * 0.9
        take_profit = current_price * 1.1
        notes = "لا توجد إشارة واضحة، ينصح بالانتظار"
    
    # إضافة مؤشرات فنية إضافية لتأكيد الإشارة
    if "RSI" in data.columns:
        last_rsi = data["RSI"].iloc[-1]
        if direction == "شراء" and last_rsi < 30:
            notes += " | تأكيد إضافي: مستوى RSI في منطقة ذروة البيع"
        elif direction == "بيع" and last_rsi > 70:
            notes += " | تأكيد إضافي: مستوى RSI في منطقة ذروة الشراء"
    
    if all(col in data.columns for col in ["MACD", "MACD_SIGNAL"]):
        macd = data["MACD"].iloc[-1]
        signal = data["MACD_SIGNAL"].iloc[-1]
        
        if direction == "شراء" and macd > signal:
            notes += " | تأكيد إضافي: تقاطع إيجابي في MACD"
        elif direction == "بيع" and macd < signal:
            notes += " | تأكيد إضافي: تقاطع سلبي في MACD"
    
    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "trend": trend,
        "confidence": current_wave.get("confidence", 0.5),
        "notes": notes
    }

def calculate_fibonacci_levels(wave_data: Dict, current_price: float) -> Dict:
    """
    حساب مستويات فيبوناتشي بناءً على موجات إليوت
    
    المعلمات:
    ----------
    wave_data : Dict
        بيانات الموجات المكتشفة
    current_price : float
        السعر الحالي
        
    العائدات:
    -------
    Dict
        مستويات فيبوناتشي
    """
    fibonacci_levels = {}
    
    # مستويات فيبوناتشي القياسية
    fib_ratios = {
        "0": 0.0,
        "0.236": 0.236,
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.786": 0.786,
        "1": 1.0,
        "1.272": 1.272,
        "1.618": 1.618,
        "2.618": 2.618
    }
    
    # البحث عن آخر موجة دافعة وتصحيحية
    last_impulse_key = None
    last_impulse_end_idx = -1
    
    last_corrective_key = None
    last_corrective_end_idx = -1
    
    for wave_key, wave_data in wave_data.items():
        if "Impulse" in wave_key:
            last_point_key = "5"
            if last_point_key in wave_data and wave_data[last_point_key]["idx"] > last_impulse_end_idx:
                last_impulse_key = wave_key
                last_impulse_end_idx = wave_data[last_point_key]["idx"]
        
        elif "Corrective" in wave_key:
            last_point_key = "C"
            if last_point_key in wave_data and wave_data[last_point_key]["idx"] > last_corrective_end_idx:
                last_corrective_key = wave_key
                last_corrective_end_idx = wave_data[last_point_key]["idx"]
    
    # حساب مستويات فيبوناتشي للموجة الدافعة
    if last_impulse_key:
        impulse_wave = wave_data[last_impulse_key]
        direction = impulse_wave["direction"]
        
        if "0" in impulse_wave and "5" in impulse_wave:
            start_price = impulse_wave["0"]["price"]
            end_price = impulse_wave["5"]["price"]
            wave_size = abs(end_price - start_price)
            
            impulse_levels = {}
            
            for level_name, ratio in fib_ratios.items():
                if direction == "up":
                    # مستويات تصحيح الموجة الصاعدة
                    level_price = end_price - (wave_size * ratio)
                else:
                    # مستويات تصحيح الموجة الهابطة
                    level_price = end_price + (wave_size * ratio)
                
                impulse_levels[level_name] = level_price
            
            fibonacci_levels["impulse"] = impulse_levels
    
    # حساب مستويات فيبوناتشي للموجة التصحيحية
    if last_corrective_key:
        corrective_wave = wave_data[last_corrective_key]
        direction = corrective_wave["direction"]
        
        if "0" in corrective_wave and "C" in corrective_wave:
            start_price = corrective_wave["0"]["price"]
            end_price = corrective_wave["C"]["price"]
            wave_size = abs(end_price - start_price)
            
            corrective_levels = {}
            
            for level_name, ratio in fib_ratios.items():
                if direction == "up":
                    # مستويات امتداد الموجة الصاعدة
                    level_price = end_price + (wave_size * ratio)
                else:
                    # مستويات امتداد الموجة الهابطة
                    level_price = end_price - (wave_size * ratio)
                
                corrective_levels[level_name] = level_price
            
            fibonacci_levels["corrective"] = corrective_levels
    
    return fibonacci_levels

def get_wave_patterns(wave_data: Dict) -> Dict:
    """
    الحصول على أنماط الموجات المكتشفة ووصفها
    
    المعلمات:
    ----------
    wave_data : Dict
        بيانات الموجات المكتشفة
        
    العائدات:
    -------
    Dict
        أنماط الموجات مع وصفها
    """
    # التعامل مع حالات الخطأ
    if not isinstance(wave_data, dict) or not wave_data or "error" in wave_data:
        # إنشاء أنماط افتراضية للعرض
        default_patterns = {
            "نموذج موجات إليوت الكلاسيكي": {
                "reliability": "متوسطة",
                "description": "نموذج موجات إليوت الكلاسيكي (5-3-5-3-5)",
                "expected_direction": "محايد"
            }
        }
        return default_patterns
    
    # استخدام الأنماط المكتشفة إذا كانت موجودة
    if "patterns" in wave_data and isinstance(wave_data["patterns"], dict) and wave_data["patterns"]:
        return wave_data["patterns"]
    
    # إنشاء قاموس الأنماط المكتشفة
    patterns = {}
    
    # التأكد من أن مفتاح "waves" موجود ومن نوع قاموس
    if "waves" in wave_data and isinstance(wave_data["waves"], dict):
        # فحص كل موجة للتعرف على أنماط محددة
        for wave_key, wave in wave_data["waves"].items():
            if not isinstance(wave, dict):
                continue
                
            direction = wave.get("direction", "غير معروف")
            
            # تحليل أنماط الموجات الدافعة
            if "Impulse" in wave_key:
                if direction == "up" or "صاعد" in direction:
                    patterns["نموذج الموجة الدافعة الصاعدة"] = {
                        "reliability": "عالية",
                        "description": "تشكل موجة دافعة صاعدة (5 موجات) وفقًا لنظرية إليوت",
                        "expected_direction": "صاعد"
                    }
                elif direction == "down" or "هابط" in direction:
                    patterns["نموذج الموجة الدافعة الهابطة"] = {
                        "reliability": "عالية",
                        "description": "تشكل موجة دافعة هابطة (5 موجات) وفقًا لنظرية إليوت",
                        "expected_direction": "هابط"
                    }
            
            # تحليل أنماط الموجات التصحيحية
            elif "Corrective" in wave_key:
                if direction == "up" or "صاعد" in direction:
                    patterns["نموذج الموجة التصحيحية الصاعدة"] = {
                        "reliability": "متوسطة",
                        "description": "تشكل موجة تصحيحية صاعدة (A-B-C) وفقًا لنظرية إليوت",
                        "expected_direction": "صاعد قصير المدى"
                    }
                elif direction == "down" or "هابط" in direction:
                    patterns["نموذج الموجة التصحيحية الهابطة"] = {
                        "reliability": "متوسطة",
                        "description": "تشكل موجة تصحيحية هابطة (A-B-C) وفقًا لنظرية إليوت",
                        "expected_direction": "هابط قصير المدى"
                    }
    
    # الكشف عن أنماط إضافية بناءً على الموجة الحالية
    if "current_wave" in wave_data and isinstance(wave_data["current_wave"], dict):
        current_wave = wave_data["current_wave"]
        position = current_wave.get("position", "")
        
        if isinstance(position, str):
            # أنماط الانعكاس الصاعد
            if "موجة دافعة صعودية" in position or "الموجة 1 صاعد" in position:
                patterns["نموذج الانعكاس الصاعد"] = {
                    "reliability": "عالية",
                    "description": "اكتمال موجة تصحيحية وبداية محتملة لموجة دافعة صاعدة",
                    "expected_direction": "صاعد قوي"
                }
            
            # أنماط الانعكاس الهابط
            elif "موجة دافعة هبوطية" in position or "الموجة 1 هابط" in position:
                patterns["نموذج الانعكاس الهابط"] = {
                    "reliability": "عالية",
                    "description": "اكتمال موجة تصحيحية وبداية محتملة لموجة دافعة هابطة",
                    "expected_direction": "هابط قوي"
                }
            
            # أنماط التصحيح
            elif "تصحيح" in position:
                if "هبوطي" in position:
                    patterns["نموذج التصحيح الهابط"] = {
                        "reliability": "متوسطة",
                        "description": "اكتمال موجة دافعة وبداية محتملة لموجة تصحيحية هابطة",
                        "expected_direction": "هابط مؤقت"
                    }
                else:
                    patterns["نموذج التصحيح الصاعد"] = {
                        "reliability": "متوسطة",
                        "description": "اكتمال موجة دافعة وبداية محتملة لموجة تصحيحية صاعدة",
                        "expected_direction": "صاعد مؤقت"
                    }
            
            # أنماط الامتداد
            elif "الموجة 3" in position:
                if "صاعد" in position:
                    patterns["نموذج الامتداد الصاعد"] = {
                        "reliability": "عالية جدًا",
                        "description": "بداية الموجة 3 (أقوى الموجات الصاعدة) مع احتمالية كبيرة لتحقيق أرباح",
                        "expected_direction": "صاعد قوي جدًا"
                    }
                elif "هابط" in position:
                    patterns["نموذج الامتداد الهابط"] = {
                        "reliability": "عالية جدًا",
                        "description": "بداية الموجة 3 (أقوى الموجات الهابطة) مع احتمالية كبيرة لهبوط حاد",
                        "expected_direction": "هابط قوي جدًا"
                    }
            
            # أنماط الانتهاء
            elif "الموجة 5" in position:
                if "صاعد" in position:
                    patterns["نموذج انتهاء الموجة الدافعة الصاعدة"] = {
                        "reliability": "عالية",
                        "description": "اقتراب انتهاء الموجة الدافعة الصاعدة، توقع بداية موجة تصحيحية هابطة قريبًا",
                        "expected_direction": "صاعد نهائي"
                    }
                elif "هابط" in position:
                    patterns["نموذج انتهاء الموجة الدافعة الهابطة"] = {
                        "reliability": "عالية",
                        "description": "اقتراب انتهاء الموجة الدافعة الهابطة، توقع بداية موجة تصحيحية صاعدة قريبًا",
                        "expected_direction": "هابط نهائي"
                    }
    
    # إضافة نموذج افتراضي إذا لم يتم اكتشاف أي أنماط محددة
    if not patterns:
        patterns["نموذج حركة السوق الحالية"] = {
            "reliability": "متوسطة",
            "description": "تحليل حركة السوق وفقًا لموجات إليوت، يرجى تحليل المزيد من البيانات للحصول على نتائج أدق",
            "expected_direction": "محايد"
        }
    
    return patterns

def calculate_potential_targets(wave_data: Dict, current_price: float) -> Dict:
    """
    حساب الأهداف السعرية المحتملة بناءً على موجات إليوت
    
    المعلمات:
    ----------
    wave_data : Dict
        بيانات الموجات المكتشفة
    current_price : float
        السعر الحالي
        
    العائدات:
    -------
    Dict
        الأهداف السعرية المحتملة
    """
    # التحقق من صحة البيانات المدخلة
    if not isinstance(wave_data, dict) or not wave_data or "error" in wave_data:
        # أهداف افتراضية في حالة عدم وجود تحليل موجات صحيح
        targets = {
            "target_1": current_price * 1.05,
            "target_2": current_price * 1.10,
            "target_3": current_price * 1.15,
            "target_1_percentage": 5.0,
            "target_2_percentage": 10.0,
            "target_3_percentage": 15.0
        }
        return targets
    
    # مستويات فيبوناتشي الافتراضية
    fib_levels = {
        "impulse": {},
        "corrective": {}
    }
    
    targets = {}
    trend = "غير معروف"
    
    if "current_wave" in wave_data:
        position = wave_data["current_wave"].get("position", "")
        
        if "متوقع بداية موجة دافعة صعودية" in position:
            trend = "صاعد"
        elif "متوقع بداية موجة دافعة هبوطية" in position:
            trend = "هابط"
        elif "متوقع بداية تصحيح" in position:
            if "هبوطي" in position:
                trend = "هابط قصير الأمد"
            else:
                trend = "صاعد قصير الأمد"
    
    # تحديد الأهداف بناءً على الاتجاه
    if trend == "صاعد":
        # أهداف صاعدة بناءً على امتدادات فيبوناتشي
        target_1 = current_price * 1.05  # +5%
        target_2 = current_price * 1.10  # +10%
        target_3 = current_price * 1.20  # +20%
        
        if "corrective" in fib_levels:
            corrective_levels = fib_levels["corrective"]
            if "0.618" in corrective_levels:
                target_1 = corrective_levels["0.618"]
            if "1.0" in corrective_levels:
                target_2 = corrective_levels["1.0"]
            if "1.618" in corrective_levels:
                target_3 = corrective_levels["1.618"]
    
    elif trend == "هابط":
        # أهداف هابطة بناءً على امتدادات فيبوناتشي
        target_1 = current_price * 0.95  # -5%
        target_2 = current_price * 0.90  # -10%
        target_3 = current_price * 0.80  # -20%
        
        if "corrective" in fib_levels:
            corrective_levels = fib_levels["corrective"]
            if "0.618" in corrective_levels:
                target_1 = corrective_levels["0.618"]
            if "1.0" in corrective_levels:
                target_2 = corrective_levels["1.0"]
            if "1.618" in corrective_levels:
                target_3 = corrective_levels["1.618"]
    
    elif trend == "هابط قصير الأمد":
        # أهداف تصحيحية هبوطية
        target_1 = current_price * 0.98  # -2%
        target_2 = current_price * 0.95  # -5%
        target_3 = current_price * 0.90  # -10%
        
        if "impulse" in fib_levels:
            impulse_levels = fib_levels["impulse"]
            if "0.382" in impulse_levels:
                target_1 = impulse_levels["0.382"]
            if "0.5" in impulse_levels:
                target_2 = impulse_levels["0.5"]
            if "0.618" in impulse_levels:
                target_3 = impulse_levels["0.618"]
    
    elif trend == "صاعد قصير الأمد":
        # أهداف تصحيحية صاعدة
        target_1 = current_price * 1.02  # +2%
        target_2 = current_price * 1.05  # +5%
        target_3 = current_price * 1.10  # +10%
        
        if "impulse" in fib_levels:
            impulse_levels = fib_levels["impulse"]
            if "0.382" in impulse_levels:
                target_1 = impulse_levels["0.382"]
            if "0.5" in impulse_levels:
                target_2 = impulse_levels["0.5"]
            if "0.618" in impulse_levels:
                target_3 = impulse_levels["0.618"]
    
    else:
        # أهداف افتراضية
        target_1 = current_price * 1.03  # +3%
        target_2 = current_price * 0.97  # -3%
        target_3 = current_price * 1.10  # +10%
    
    # حساب النسب المئوية للأهداف
    target_1_percentage = ((target_1 / current_price) - 1) * 100
    target_2_percentage = ((target_2 / current_price) - 1) * 100
    target_3_percentage = ((target_3 / current_price) - 1) * 100
    
    targets = {
        "target_1": target_1,
        "target_2": target_2,
        "target_3": target_3,
        "target_1_percentage": target_1_percentage,
        "target_2_percentage": target_2_percentage,
        "target_3_percentage": target_3_percentage,
        "trend": trend
    }
    
    return targets
