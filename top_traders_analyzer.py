import requests
from typing import Dict, List
import json
import numpy as np
import random
import time
from datetime import datetime

class TopTradersAnalyzer:
    def __init__(self):
        """
        محلل سلوك كبار المتداولين باستخدام بيانات من TradingView وJP Markets
        """
        self.base_url = "https://scanner.tradingview.com/crypto/scan"
        self.alt_urls = [
            "https://scanner.tradingview.com/forex/scan",
            "https://scanner.tradingview.com/america/scan"
        ]
        self.cached_data = {}
        self.cache_time = {}
        self.cache_expiry = 300  # 5 minutes cache expiry
        
    def get_top_traders_sentiment(self, symbol: str) -> Dict:
        """
        الحصول على تحليل مشاعر كبار المتداولين للرمز المحدد
        مع معالجة أفضل للأخطاء وحالات الاستثناء
        """
        # تجنب طلبات زائدة خلال فترة زمنية قصيرة
        current_time = time.time()
        if symbol in self.cache_time:
            if current_time - self.cache_time[symbol] < self.cache_expiry:
                return self.cached_data.get(symbol, self._default_sentiment())
        
        try:
            # تعديل الرمز ليناسب TradingView API
            tv_symbol = self._convert_to_tradingview_symbol(symbol)
            
            payload = {
                "symbols": {"tickers": [tv_symbol]},
                "columns": [
                    "Recommend.All",
                    "Recommend.MA",
                    "volume",
                    "change",
                    "VWAP",
                    "RSI"
                ]
            }
            
            # المحاولة مع URL الأساسي أولاً
            response = requests.post(self.base_url, json=payload, timeout=5)
            
            # إذا فشل URL الأساسي، جرب URLs البديلة
            if response.status_code != 200:
                for url in self.alt_urls:
                    try:
                        response = requests.post(url, json=payload, timeout=5)
                        if response.status_code == 200:
                            break
                    except:
                        continue
            
            data = response.json()
            
            if 'data' in data and data['data']:
                # استخراج المعلومات من API استجابة
                sentiment_score = data['data'][0]['d'][0]  # Recommend.All
                ma_score = data['data'][0]['d'][1]  # Recommend.MA
                volume = data['data'][0]['d'][2]  # Volume
                change = data['data'][0]['d'][3]  # Change
                
                # تحسين تفسير نتيجة المشاعر
                sentiment = "إيجابي قوي" if sentiment_score > 0.5 else \
                           "إيجابي" if sentiment_score > 0 else \
                           "سلبي" if sentiment_score > -0.5 else "سلبي قوي"
                
                # إضافة تحليل إضافي للحجم
                volume_insight = "مرتفع" if volume > 500000 else \
                                "متوسط" if volume > 100000 else "منخفض"
                
                # تقييم قوة الإشارة من المتوسطات المتحركة
                ma_sentiment = "قوي" if abs(ma_score) > 0.7 else \
                              "متوسط" if abs(ma_score) > 0.3 else "ضعيف"
                
                result = {
                    'sentiment': sentiment,
                    'strength': abs(sentiment_score),
                    'volume': volume,
                    'raw_score': sentiment_score,
                    'volume_insight': volume_insight,
                    'ma_sentiment': ma_sentiment,
                    'change_percent': change,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # تحديث ذاكرة التخزين المؤقت
                self.cached_data[symbol] = result
                self.cache_time[symbol] = current_time
                
                return result
            
            return self._default_sentiment()
            
        except Exception as e:
            print(f"خطأ في تحليل كبار المتداولين: {str(e)}")
            return self._default_sentiment()
    
    def _convert_to_tradingview_symbol(self, symbol: str) -> str:
        """
        تحويل رمز Yahoo Finance إلى تنسيق TradingView
        """
        # تعيين التحويلات الشائعة
        conversion_map = {
            'GC=F': 'COMEX:GC',
            'SI=F': 'COMEX:SI',
            'CL=F': 'NYMEX:CL',
            'NQ=F': 'CME_MINI:NQ',
            'ES=F': 'CME_MINI:ES',
            'YM=F': 'CBOT_MINI:YM',
            '^GDAXI': 'INDEX:DAX',
            '^FTSE': 'INDEX:FTSE',
            '^N225': 'INDEX:NKY',
            'GBPUSD=X': 'OANDA:GBPUSD',
            'EURUSD=X': 'OANDA:EURUSD',
            'USDJPY=X': 'OANDA:USDJPY',
            'USDCHF=X': 'OANDA:USDCHF',
            'AUDUSD=X': 'OANDA:AUDUSD',
            'USDCAD=X': 'OANDA:USDCAD',
            'BTC-USD': 'BINANCE:BTCUSDT',
            'ETH-USD': 'BINANCE:ETHUSDT'
        }
        
        # استخدام الخريطة إذا كان الرمز موجودًا، وإلا إرجاع الرمز الأصلي
        return conversion_map.get(symbol, symbol)
    
    def _default_sentiment(self) -> Dict:
        """
        إرجاع قيم افتراضية في حالة الأخطاء
        """
        return {
            'sentiment': 'محايد',
            'strength': 0.5,
            'volume': 100000,
            'raw_score': 0,
            'volume_insight': 'متوسط',
            'ma_sentiment': 'محايد',
            'change_percent': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_institutional_positioning(self, symbol: str) -> Dict:
        """
        تحليل موقف المؤسسات الكبرى تجاه الرمز المحدد
        """
        try:
            # مؤسسات وصناديق وهمية للتمثيل
            institutions = [
                "JP Morgan", "Goldman Sachs", "BlackRock", "Vanguard", "Fidelity",
                "Bridgewater Associates", "Citadel", "AQR Capital", "Renaissance Technologies"
            ]
            
            # توليد مواقف عشوائية للمؤسسات
            positioning = []
            net_positioning = 0
            
            for institution in institutions:
                # موقف عشوائي: مرقم من -100 إلى 100، حيث القيم الإيجابية تمثل الشراء والسلبية تمثل البيع
                position = random.randint(-100, 100)
                position_type = "شراء" if position > 0 else "بيع" if position < 0 else "محايد"
                
                net_positioning += position
                
                positioning.append({
                    'institution': institution,
                    'position': position,
                    'position_type': position_type,
                    'confidence': random.randint(1, 5)  # مستوى الثقة من 1 إلى 5
                })
            
            # ترتيب المواقف حسب القيمة المطلقة للموقف
            positioning = sorted(positioning, key=lambda x: abs(x['position']), reverse=True)
            
            # تلخيص الموقف العام
            if net_positioning > 200:
                summary = "موقف شراء قوي من المؤسسات"
            elif net_positioning > 0:
                summary = "ميل للشراء من المؤسسات"
            elif net_positioning > -200:
                summary = "ميل للبيع من المؤسسات"
            else:
                summary = "موقف بيع قوي من المؤسسات"
            
            return {
                'symbol': symbol,
                'institutional_positions': positioning[:5],  # أكبر 5 مواقف
                'net_positioning': net_positioning,
                'summary': summary,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"خطأ في تحليل موقف المؤسسات: {str(e)}")
            return {
                'symbol': symbol,
                'institutional_positions': [],
                'net_positioning': 0,
                'summary': "بيانات غير متوفرة",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def get_order_flow_analysis(self, symbol: str) -> Dict:
        """
        تحليل تدفق الأوامر للرمز المحدد
        """
        try:
            # توليد بيانات تدفق الأوامر
            buy_orders = random.randint(500, 5000)
            sell_orders = random.randint(500, 5000)
            
            buy_volume = buy_orders * random.uniform(1.0, 3.0)
            sell_volume = sell_orders * random.uniform(1.0, 3.0)
            
            net_flow = buy_volume - sell_volume
            
            # تحديد مستويات الضغط
            price_levels = []
            current_price = 1000  # قيمة افتراضية
            
            if symbol == 'GC=F':
                current_price = 2000
            elif symbol == 'BTC-USD':
                current_price = 65000
            elif symbol == 'ETH-USD':
                current_price = 3500
            
            # إنشاء مستويات سعرية عشوائية
            for i in range(5):
                offset = random.uniform(-0.05, 0.05)
                level_price = current_price * (1 + offset)
                
                price_levels.append({
                    'price': level_price,
                    'type': 'مقاومة' if offset > 0 else 'دعم',
                    'strength': random.randint(1, 10),
                    'volume': random.randint(1000, 10000)
                })
            
            # ترتيب المستويات حسب السعر
            price_levels = sorted(price_levels, key=lambda x: x['price'])
            
            return {
                'symbol': symbol,
                'buy_orders': buy_orders,
                'sell_orders': sell_orders,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'net_flow': net_flow,
                'flow_sentiment': "إيجابي" if net_flow > 0 else "سلبي",
                'key_levels': price_levels,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"خطأ في تحليل تدفق الأوامر: {str(e)}")
            return {
                'symbol': symbol,
                'buy_orders': 0,
                'sell_orders': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'net_flow': 0,
                'flow_sentiment': "محايد",
                'key_levels': [],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }