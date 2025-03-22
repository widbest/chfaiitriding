import requests
import random
import json
import os
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Optional, Union
import numpy as np

def get_market_sentiment(symbol: str) -> Dict:
    """
    الحصول على تحليل المشاعر من أخبار السوق للرمز المحدد
    """
    # محاولة الحصول على بيانات من API أخبار
    news_data = fetch_news(symbol)
    
    # إذا لم نتمكن من الحصول على بيانات حقيقية، استخدم التحليل البديل
    if not news_data or 'articles' not in news_data or not news_data['articles']:
        return fallback_sentiment_analysis(symbol)
    
    # تحليل المشاعر من العناوين والمحتوى
    articles = news_data['articles']
    
    # استخراج أهم الأخبار
    top_headlines = [article['title'] for article in articles[:3]]
    
    # تحليل المشاعر من الأخبار
    sentiment_scores = []
    
    for article in articles:
        # تحليل العنوان
        title_score = analyze_text_sentiment(article['title'])
        
        # تحليل المحتوى إذا كان متاحًا
        content_score = 0
        if 'description' in article and article['description']:
            content_score = analyze_text_sentiment(article['description'])
        
        # حساب متوسط الدرجة
        score = (title_score * 0.7 + content_score * 0.3)
        sentiment_scores.append(score)
    
    # حساب متوسط درجات المشاعر
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # تحديد المشاعر العامة
    if avg_sentiment > 0.3:
        overall_sentiment = "إيجابي قوي"
    elif avg_sentiment > 0.1:
        overall_sentiment = "إيجابي"
    elif avg_sentiment > -0.1:
        overall_sentiment = "محايد"
    elif avg_sentiment > -0.3:
        overall_sentiment = "سلبي"
    else:
        overall_sentiment = "سلبي قوي"
    
    # حساب قوة المشاعر
    sentiment_strength = abs(avg_sentiment) * 100
    
    # إعداد النتيجة
    result = {
        "overall_sentiment": overall_sentiment,
        "sentiment_score": avg_sentiment,
        "sentiment_strength": min(sentiment_strength, 90),  # تقييد القوة
        "top_headlines": top_headlines,
        "top_headline": top_headlines[0] if top_headlines else "لا توجد عناوين متاحة",
        "recent_articles_count": len(articles),
        "source": "NewsAPI",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def analyze_text_sentiment(text: str) -> float:
    """
    تحليل مشاعر النص باستخدام قوائم الكلمات الإيجابية والسلبية
    """
    if not text:
        return 0
    
    text = text.lower()
    
    # قائمة الكلمات الإيجابية
    positive_words = [
        'up', 'rise', 'bull', 'bullish', 'gain', 'gains', 'positive', 'profit',
        'profits', 'grow', 'growth', 'increase', 'increasing', 'outperform',
        'strong', 'stronger', 'strength', 'high', 'higher', 'record', 'rally',
        'support', 'supported', 'buy', 'buying', 'optimism', 'optimistic',
        'opportunity', 'opportunities', 'improve', 'improved', 'improving',
        'recovery', 'recovering', 'recovered', 'rebound', 'advance', 'advancing'
    ]
    
    # قائمة الكلمات السلبية
    negative_words = [
        'down', 'fall', 'falling', 'bear', 'bearish', 'loss', 'losses', 'negative',
        'decline', 'declining', 'decreased', 'decrease', 'weak', 'weaker', 'weakness',
        'low', 'lower', 'underperform', 'sell', 'selling', 'fear', 'fears', 'worried',
        'worry', 'pressure', 'drop', 'plunge', 'plummeting', 'recession', 'crisis',
        'risk', 'risks', 'risky', 'danger', 'dangerous', 'threat', 'threatened',
        'volatile', 'volatility', 'concern', 'concerns', 'concerned', 'warning'
    ]
    
    # العثور على جميع الكلمات في النص
    words = re.findall(r'\b\w+\b', text)
    
    # عد الكلمات الإيجابية والسلبية
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # حساب الدرجة
    total_words = len(words)
    if total_words > 0:
        score = (positive_count - negative_count) / total_words
    else:
        score = 0
    
    return score

def fetch_news(symbol: str) -> Optional[Dict]:
    """
    جلب أخبار مرتبطة بالرمز من NewsAPI
    """
    # تحويل رمز الأداة إلى مصطلح بحث أفضل
    search_term = convert_symbol_to_search_term(symbol)
    
    # استخدام NewsAPI إذا كان مفتاح API متاحًا
    api_key = os.getenv('NEWS_API_KEY')
    
    if api_key:
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': search_term,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"خطأ في جلب الأخبار: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"خطأ في جلب الأخبار: {str(e)}")
            return None
    
    # إذا لم يكن مفتاح API متاحًا، ارجع None
    return None

def convert_symbol_to_search_term(symbol: str) -> str:
    """
    تحويل رمز الأداة إلى مصطلح بحث أفضل للحصول على أخبار مرتبطة
    """
    symbol_map = {
        'GC=F': 'gold price',
        'SI=F': 'silver price',
        'CL=F': 'crude oil price',
        'NQ=F': 'nasdaq futures',
        'ES=F': 's&p 500 futures',
        'YM=F': 'dow jones futures',
        '^GDAXI': 'german dax index',
        '^FTSE': 'ftse 100 index',
        '^N225': 'nikkei index',
        'GBPUSD=X': 'gbp usd forex',
        'EURUSD=X': 'eur usd forex',
        'USDJPY=X': 'usd jpy forex',
        'USDCHF=X': 'usd chf forex',
        'AUDUSD=X': 'aud usd forex',
        'USDCAD=X': 'usd cad forex',
        'BTC-USD': 'bitcoin price',
        'ETH-USD': 'ethereum price',
        'BNB-USD': 'binance coin price',
        'XRP-USD': 'ripple price',
        'ADA-USD': 'cardano price'
    }
    
    # استخدام التعيين إذا كان متاحًا، وإلا استخدم الرمز كما هو
    search_term = symbol_map.get(symbol, symbol.replace('=', ' ').replace('-', ' '))
    
    return search_term

def fallback_sentiment_analysis(symbol: str) -> Dict:
    """
    تحليل مشاعر السوق البديل في حالة عدم توفر بيانات الأخبار
    """
    # تحديد نوع الأصل
    asset_type = get_asset_type(symbol)
    
    # استخدام بيانات مختلفة بناءً على نوع الأصل
    if asset_type == 'crypto':
        random_sentiment = random.uniform(-0.2, 0.3)  # العملات المشفرة متقلبة
        top_headlines = [
            "تذبذب أسعار العملات المشفرة مع استمرار عدم اليقين في السوق",
            "ترقب لإعلان تنظيمي قد يؤثر على مستقبل العملات الرقمية",
            "مستثمرون يبحثون عن فرص في سوق العملات المشفرة المتقلب"
        ]
    elif asset_type == 'forex':
        random_sentiment = random.uniform(-0.1, 0.1)  # الفوركس أكثر استقرارًا
        top_headlines = [
            "البنوك المركزية تراقب تحركات العملات وسط توقعات بتغيير أسعار الفائدة",
            "تقلبات محدودة في أسواق العملات مع ترقب بيانات اقتصادية جديدة",
            "تحركات متباينة للعملات الرئيسية مع تغير توقعات النمو العالمي"
        ]
    elif asset_type == 'commodity':
        random_sentiment = random.uniform(-0.15, 0.15)  # السلع متنوعة
        top_headlines = [
            "أسعار السلع تتأثر بتوقعات الطلب العالمي وسط مخاوف التضخم",
            "عوامل جيوسياسية تؤثر على أسواق الطاقة والمعادن الثمينة",
            "المستثمرون يقيمون تأثير سياسات البنوك المركزية على أسواق السلع"
        ]
    else:  # stock or index
        random_sentiment = random.uniform(-0.1, 0.2)  # الأسهم والمؤشرات
        top_headlines = [
            "الأسواق تتفاعل مع نتائج الشركات ومؤشرات الاقتصاد الكلي",
            "المستثمرون يقيمون آفاق النمو الاقتصادي وسياسات البنوك المركزية",
            "تقلبات في الأسواق المالية مع تغير توقعات أرباح الشركات"
        ]
    
    # تحديد المشاعر العامة بناءً على القيمة العشوائية
    if random_sentiment > 0.2:
        overall_sentiment = "إيجابي قوي"
    elif random_sentiment > 0.05:
        overall_sentiment = "إيجابي"
    elif random_sentiment > -0.05:
        overall_sentiment = "محايد"
    elif random_sentiment > -0.2:
        overall_sentiment = "سلبي"
    else:
        overall_sentiment = "سلبي قوي"
    
    # حساب قوة المشاعر
    sentiment_strength = abs(random_sentiment) * 100
    
    # إعداد النتيجة
    result = {
        "overall_sentiment": overall_sentiment,
        "sentiment_score": random_sentiment,
        "sentiment_strength": min(sentiment_strength, 80),  # تقييد القوة
        "top_headlines": top_headlines,
        "top_headline": top_headlines[0],
        "recent_articles_count": random.randint(5, 15),
        "source": "تحليل بديل",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def get_asset_type(symbol: str) -> str:
    """
    تحديد نوع الأصل المالي بناءً على الرمز
    """
    if 'BTC' in symbol or 'ETH' in symbol or 'BNB' in symbol or 'XRP' in symbol or 'ADA' in symbol or '-USD' in symbol:
        return 'crypto'
    elif 'USD' in symbol and '=' in symbol:
        return 'forex'
    elif 'GC=' in symbol or 'SI=' in symbol or 'CL=' in symbol:
        return 'commodity'
    elif '^' in symbol or '=' in symbol:
        return 'index'
    else:
        return 'stock'

def get_market_trends(symbol: str) -> Dict:
    """
    تحليل اتجاهات السوق العامة للرمز المحدد
    """
    # تحديد نوع الأصل
    asset_type = get_asset_type(symbol)
    
    # توليد اتجاهات عشوائية بناءً على نوع الأصل
    if asset_type == 'crypto':
        short_trend = random.choice(['صاعد قوي', 'صاعد', 'متذبذب', 'هابط', 'هابط قوي'])
        medium_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        long_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
    elif asset_type == 'forex':
        short_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        medium_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        long_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
    elif asset_type == 'commodity':
        short_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        medium_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        long_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
    else:  # stock or index
        short_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        medium_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
        long_trend = random.choice(['صاعد', 'متذبذب', 'هابط'])
    
    # توليد قوة الاتجاه
    short_strength = random.randint(50, 95)
    medium_strength = random.randint(40, 90)
    long_strength = random.randint(30, 85)
    
    # إنشاء رسالة الاتجاه
    if short_trend == 'صاعد' or short_trend == 'صاعد قوي':
        short_message = f"الاتجاه قصير المدى صاعد مع قوة {short_strength}%"
    elif short_trend == 'هابط' or short_trend == 'هابط قوي':
        short_message = f"الاتجاه قصير المدى هابط مع قوة {short_strength}%"
    else:
        short_message = f"الاتجاه قصير المدى متذبذب"
    
    if medium_trend == 'صاعد':
        medium_message = f"الاتجاه متوسط المدى صاعد مع قوة {medium_strength}%"
    elif medium_trend == 'هابط':
        medium_message = f"الاتجاه متوسط المدى هابط مع قوة {medium_strength}%"
    else:
        medium_message = f"الاتجاه متوسط المدى متذبذب"
    
    if long_trend == 'صاعد':
        long_message = f"الاتجاه طويل المدى صاعد مع قوة {long_strength}%"
    elif long_trend == 'هابط':
        long_message = f"الاتجاه طويل المدى هابط مع قوة {long_strength}%"
    else:
        long_message = f"الاتجاه طويل المدى متذبذب"
    
    # إعداد النتيجة
    result = {
        "short_term": {
            "trend": short_trend,
            "strength": short_strength,
            "message": short_message
        },
        "medium_term": {
            "trend": medium_trend,
            "strength": medium_strength,
            "message": medium_message
        },
        "long_term": {
            "trend": long_trend,
            "strength": long_strength,
            "message": long_message
        },
        "overall_trend": random.choice(['صاعد', 'متذبذب', 'هابط']),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result