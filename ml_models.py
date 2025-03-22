import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    تحضير الميزات للتدريب والتنبؤ
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
        
    العائدات:
    -------
    Tuple[pd.DataFrame, str]
        إطار البيانات المحضر واسم المؤشر
    """
    # نسخ البيانات لتجنب التعديل المباشر
    df = data.copy()
    
    # استخراج اسم المؤشر إذا كان متاحًا
    symbol = getattr(df, 'name', 'unknown')
    
    # التأكد من وجود أعمدة التاريخ والسعر
    if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = df.index
    
    # حساب ميزات الاتجاه
    df['price_lag1'] = df['Close'].shift(1)
    df['price_lag2'] = df['Close'].shift(2)
    df['price_lag3'] = df['Close'].shift(3)
    df['price_lag5'] = df['Close'].shift(5)
    df['price_lag10'] = df['Close'].shift(10)
    
    # حساب التغييرات في أسعار الإغلاق
    df['price_change'] = df['Close'].pct_change()
    df['price_change_lag1'] = df['price_change'].shift(1)
    df['price_change_lag2'] = df['price_change'].shift(2)
    
    # حساب المتوسطات المتحركة إذا لم تكن موجودة بالفعل
    if 'SMA20' not in df.columns:
        df['SMA20'] = df['Close'].rolling(window=20).mean()
    if 'SMA50' not in df.columns:
        df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # حساب المسافات من المتوسطات المتحركة
    df['dist_from_sma20'] = (df['Close'] - df['SMA20']) / df['SMA20']
    df['dist_from_sma50'] = (df['Close'] - df['SMA50']) / df['SMA50']
    
    # إضافة مؤشر القوة النسبية إذا لم يكن موجودًا
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # إضافة مؤشر التقلب
    df['volatility'] = df['Close'].rolling(window=20).std() / df['Close']
    
    # استخراج ميزات زمنية
    if isinstance(df['Date'].iloc[0], datetime) or isinstance(df['Date'].iloc[0], pd.Timestamp):
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
    
    # ملء القيم المفقودة
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # حذف صفوف تحتوي على قيم مفقودة
    df = df.dropna()
    
    return df, symbol

def select_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    اختيار الميزات المناسبة وتقسيم البيانات
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات المحضر
        
    العائدات:
    -------
    Tuple[pd.DataFrame, pd.Series]
        ميزات X والمتغير المستهدف y
    """
    # اختيار الميزات ذات الصلة
    feature_columns = [
        'price_lag1', 'price_lag2', 'price_lag3', 'price_lag5',
        'price_change_lag1', 'price_change_lag2',
        'dist_from_sma20', 'dist_from_sma50',
        'volatility'
    ]
    
    # إضافة المؤشرات الفنية إذا كانت متاحة
    for col in ['RSI', 'MACD', 'ATR', 'OBV', 'CCI']:
        if col in data.columns:
            feature_columns.append(col)
    
    # إضافة الميزات الزمنية إذا كانت متاحة
    for col in ['day_of_week', 'day_of_month', 'month']:
        if col in data.columns:
            feature_columns.append(col)
    
    # التحقق من وجود جميع الميزات في البيانات
    feature_columns = [col for col in feature_columns if col in data.columns]
    
    # تقسيم الميزات والمتغير المستهدف
    X = data[feature_columns]
    y = data['Close']
    
    return X, y

def train_model(data: pd.DataFrame) -> object:
    """
    تدريب نموذج للتنبؤ بحركة الأسعار المستقبلية
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
        
    العائدات:
    -------
    object
        النموذج المدرب
    """
    try:
        # تحضير البيانات
        prepared_data, symbol = prepare_features(data)
        
        # اختيار الميزات
        X, y = select_features(prepared_data)
        
        # تقسيم البيانات إلى مجموعات تدريب واختبار
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # تدريب النموذج
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # تقييم النموذج
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"تم تدريب النموذج بنجاح لـ {symbol}")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # حفظ النموذج
        save_model(model, symbol)
        
        return model
        
    except Exception as e:
        print(f"خطأ في تدريب النموذج: {str(e)}")
        # تجربة نموذج أبسط في حالة فشل النموذج الأول
        try:
            # تحضير البيانات
            prepared_data, symbol = prepare_features(data)
            
            # اختيار الميزات
            X, y = select_features(prepared_data)
            
            # تدريب نموذج خطي بسيط
            model = LinearRegression()
            model.fit(X, y)
            
            # حفظ النموذج
            save_model(model, symbol)
            
            return model
            
        except Exception as e2:
            print(f"خطأ في تدريب النموذج البديل: {str(e2)}")
            return None

def save_model(model: object, symbol: str) -> None:
    """
    حفظ النموذج المدرب في ملف
    
    المعلمات:
    ----------
    model : object
        النموذج المدرب
    symbol : str
        رمز الأداة المالية
    """
    try:
        # إنشاء مجلد للنماذج إذا لم يكن موجودًا
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # حفظ النموذج
        model_path = f'models/{symbol}_model.joblib'
        joblib.dump(model, model_path)
        
        print(f"تم حفظ النموذج في {model_path}")
        
    except Exception as e:
        print(f"خطأ في حفظ النموذج: {str(e)}")

def load_model_if_exists(symbol: str) -> Optional[object]:
    """
    تحميل النموذج المحفوظ إذا كان موجودًا
    
    المعلمات:
    ----------
    symbol : str
        رمز الأداة المالية
        
    العائدات:
    -------
    Optional[object]
        النموذج المحمل أو None إذا لم يكن موجودًا
    """
    try:
        model_path = f'models/{symbol}_model.joblib'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"تم تحميل النموذج من {model_path}")
            return model
        else:
            print(f"لم يتم العثور على نموذج محفوظ لـ {symbol}")
            return None
            
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {str(e)}")
        return None

def predict_next_movement(model: object, data: pd.DataFrame, forecast_periods: int = 5) -> List[float]:
    """
    التنبؤ بحركة الأسعار المستقبلية
    
    المعلمات:
    ----------
    model : object
        النموذج المدرب
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
    forecast_periods : int
        عدد الفترات المستقبلية للتنبؤ
        
    العائدات:
    -------
    List[float]
        قائمة بالأسعار المتوقعة
    """
    if model is None:
        # إرجاع تنبؤات افتراضية بناءً على الاتجاه الحالي
        last_prices = data['Close'].iloc[-5:].values
        avg_change = np.mean(np.diff(last_prices) / last_prices[:-1])
        
        predictions = []
        last_price = data['Close'].iloc[-1]
        
        for _ in range(forecast_periods):
            next_price = last_price * (1 + avg_change)
            predictions.append(next_price)
            last_price = next_price
        
        return predictions
    
    try:
        # تحضير البيانات
        prepared_data, _ = prepare_features(data)
        
        # اختيار الميزات
        X, _ = select_features(prepared_data)
        
        # الحصول على أحدث بيانات للتنبؤ
        latest_features = X.iloc[-1:].copy()
        
        predictions = []
        last_close = data['Close'].iloc[-1]
        
        # التنبؤ لكل فترة
        for i in range(forecast_periods):
            # التنبؤ بالسعر التالي
            next_price = model.predict(latest_features)[0]
            predictions.append(next_price)
            
            # تحديث الميزات للفترة التالية
            price_change = (next_price - last_close) / last_close
            
            # تحديث البيانات
            if 'price_lag1' in latest_features.columns:
                latest_features['price_lag1'] = last_close
            if 'price_lag2' in latest_features.columns:
                latest_features['price_lag2'] = latest_features['price_lag1'].iloc[0]
            if 'price_lag3' in latest_features.columns:
                latest_features['price_lag3'] = latest_features['price_lag2'].iloc[0]
            if 'price_lag5' in latest_features.columns and i >= 2:
                latest_features['price_lag5'] = latest_features['price_lag3'].iloc[0]
            
            if 'price_change_lag1' in latest_features.columns:
                latest_features['price_change_lag1'] = price_change
            if 'price_change_lag2' in latest_features.columns:
                latest_features['price_change_lag2'] = latest_features['price_change_lag1'].iloc[0]
            
            # تحديث المتوسطات المتحركة (تقريب بسيط)
            if 'dist_from_sma20' in latest_features.columns:
                sma20 = (latest_features['price_lag1'].iloc[0] * 19 + next_price) / 20
                latest_features['dist_from_sma20'] = (next_price - sma20) / sma20
            
            if 'dist_from_sma50' in latest_features.columns:
                sma50 = (latest_features['price_lag1'].iloc[0] * 49 + next_price) / 50
                latest_features['dist_from_sma50'] = (next_price - sma50) / sma50
            
            # تحديث السعر الأخير للفترة التالية
            last_close = next_price
        
        return predictions
        
    except Exception as e:
        print(f"خطأ في التنبؤ: {str(e)}")
        
        # إرجاع تنبؤات افتراضية في حالة الخطأ
        last_prices = data['Close'].iloc[-5:].values
        avg_change = np.mean(np.diff(last_prices) / last_prices[:-1])
        
        predictions = []
        last_price = data['Close'].iloc[-1]
        
        for _ in range(forecast_periods):
            next_price = last_price * (1 + avg_change)
            predictions.append(next_price)
            last_price = next_price
        
        return predictions

def evaluate_model_performance(model: object, data: pd.DataFrame) -> Dict:
    """
    تقييم أداء النموذج
    
    المعلمات:
    ----------
    model : object
        النموذج المدرب
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
        
    العائدات:
    -------
    Dict
        مقاييس الأداء
    """
    if model is None:
        return {
            'mse': 0,
            'mae': 0,
            'r2': 0,
            'accuracy': 0,
            'direction_accuracy': 0
        }
    
    try:
        # تحضير البيانات
        prepared_data, _ = prepare_features(data)
        
        # اختيار الميزات
        X, y = select_features(prepared_data)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        
        # التنبؤ على بيانات الاختبار
        y_pred = model.predict(X_test)
        
        # حساب مقاييس الخطأ
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # حساب دقة اتجاه الحركة
        actual_direction = np.sign(y_test.values[1:] - y_test.values[:-1])
        pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
        
    except Exception as e:
        print(f"خطأ في تقييم النموذج: {str(e)}")
        return {
            'mse': 0,
            'mae': 0,
            'r2': 0,
            'direction_accuracy': 0
        }

def get_model_feature_importance(model: object, data: pd.DataFrame) -> Dict:
    """
    الحصول على أهمية الميزات في النموذج
    
    المعلمات:
    ----------
    model : object
        النموذج المدرب
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
        
    العائدات:
    -------
    Dict
        أهمية الميزات
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return {}
    
    try:
        # تحضير البيانات
        prepared_data, _ = prepare_features(data)
        
        # اختيار الميزات
        X, _ = select_features(prepared_data)
        
        # الحصول على أهمية الميزات
        feature_importance = model.feature_importances_
        
        # ترتيب الميزات حسب الأهمية
        importance_dict = {}
        for i, feature in enumerate(X.columns):
            importance_dict[feature] = feature_importance[i]
        
        # ترتيب القاموس تنازليًا حسب الأهمية
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
        
    except Exception as e:
        print(f"خطأ في استخراج أهمية الميزات: {str(e)}")
        return {}

def predict_price_range(model: object, data: pd.DataFrame, days_ahead: int = 5, confidence: float = 0.8) -> Dict:
    """
    توقع نطاق السعر المستقبلي مع فترة ثقة
    
    المعلمات:
    ----------
    model : object
        النموذج المدرب
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار والمؤشرات الفنية
    days_ahead : int
        عدد الأيام المستقبلية للتوقع
    confidence : float
        مستوى الثقة (0-1)
        
    العائدات:
    -------
    Dict
        معلومات نطاق السعر المتوقع
    """
    # الحصول على توقعات النقطة
    point_predictions = predict_next_movement(model, data, days_ahead)
    
    # حساب التقلب التاريخي
    historical_volatility = data['Close'].pct_change().std()
    
    # حساب نطاق التوقع لكل يوم
    prediction_ranges = []
    
    for i, pred in enumerate(point_predictions):
        # زيادة عدم اليقين مع المستقبل البعيد
        day_factor = (i + 1) ** 0.5
        
        # حساب حدود النطاق
        z_score = 1.96  # 95% فترة ثقة
        margin = pred * historical_volatility * day_factor * z_score
        
        lower_bound = pred - margin
        upper_bound = pred + margin
        
        prediction_ranges.append({
            'day': i + 1,
            'prediction': pred,
            'lower_bound': max(0, lower_bound),  # لضمان عدم وجود أسعار سالبة
            'upper_bound': upper_bound,
            'range_width': upper_bound - max(0, lower_bound)
        })
    
    # حساب الاتجاه العام
    if len(point_predictions) > 1:
        trend = "صاعد" if point_predictions[-1] > point_predictions[0] else "هابط" if point_predictions[-1] < point_predictions[0] else "محايد"
        trend_strength = abs(point_predictions[-1] - point_predictions[0]) / point_predictions[0]
    else:
        trend = "غير معروف"
        trend_strength = 0
    
    return {
        'prediction_ranges': prediction_ranges,
        'trend': trend,
        'trend_strength': trend_strength,
        'confidence_level': confidence
    }
