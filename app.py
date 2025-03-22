import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json

# استيراد الوحدات المخصصة
from market_data import fetch_market_data, prepare_data, get_valid_symbols
from technical_indicators import add_indicators, get_available_indicators
from elliott_wave_analyzer import identify_elliott_waves, get_wave_patterns, calculate_potential_targets
from backtesting import backtest_strategy, calculate_performance_metrics
from ml_models import predict_next_movement, train_model, load_model_if_exists
from sentiment_analyzer import get_market_sentiment
from top_traders_analyzer import TopTradersAnalyzer
from utils import format_number, calculate_risk_reward_ratio, get_market_trends, get_current_market_status
from high_probability_signal import generate_high_probability_signal, format_trade_signal, validate_trading_opportunity

# إعداد الصفحة
st.set_page_config(
    page_title="محلل موجات إليوت - تحليل الأسواق المالية",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تهيئة المتغيرات في الجلسة
if 'data' not in st.session_state:
    st.session_state.data = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = None
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "1d"
if 'waves' not in st.session_state:
    st.session_state.waves = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# تعريف تابع لعرض الوقت بالتنسيق العربي
def get_arabic_time():
    now = datetime.datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"

# إنشاء كائن محلل كبار المتداولين
top_traders_analyzer = TopTradersAnalyzer()

# العنوان والمقدمة
st.title("🌊 محلل موجات إليوت")
st.subheader("تحليل الأسواق المالية باستخدام نظرية موجات إليوت")

# الشريط الجانبي
with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    
    # اختيار الرمز والإطار الزمني
    valid_symbols = get_valid_symbols()
    symbol = st.selectbox(
        "اختر الرمز المالي:",
        options=list(valid_symbols.keys()),
        format_func=lambda x: f"{x} - {valid_symbols[x]}",
        key="symbol_selector"
    )
    
    timeframe_options = {
        "1d": "يومي",
        "1h": "ساعة",
        "1m": "دقيقة",
        "1mo": "شهري",
        "3mo": "ربع سنوي",
        "6mo": "نصف سنوي",
        "1y": "سنوي",
        "2y": "سنتين"
    }
    
    timeframe = st.selectbox(
        "اختر الإطار الزمني:",
        options=list(timeframe_options.keys()),
        format_func=lambda x: timeframe_options[x],
        key="timeframe_selector"
    )
    
    # زر التحديث
    col1, col2 = st.columns(2)
    with col1:
        if st.button("تحديث البيانات", key="refresh_button"):
            st.session_state.data = None
            st.session_state.waves = None
            st.rerun()
    
    with col2:
        if st.button("تحليل الموجات", key="analyze_button"):
            with st.spinner("جاري تحليل الموجات..."):
                if st.session_state.data is not None:
                    wave_data = identify_elliott_waves(st.session_state.data)
                    st.session_state.waves = wave_data
                    st.success("تم تحليل الموجات بنجاح!")
                    st.rerun()
                else:
                    st.error("يرجى تحديث البيانات أولاً")
    
    # عرض حالة السوق
    st.header("🕒 حالة السوق")
    market_status = get_current_market_status()
    st.info(f"الجلسة الحالية: {market_status['trading_session']}")
    st.write(f"الأسواق النشطة: {', '.join(market_status['active_markets'])}")
    
    if market_status['upcoming_events']:
        st.write("أحداث قادمة:")
        for event in market_status['upcoming_events']:
            st.write(f"- {event}")
    
    st.write(f"آخر تحديث: {get_arabic_time()}")
    
    # إعدادات متقدمة
    st.header("🛠️ إعدادات متقدمة")
    wave_threshold = st.slider(
        "حساسية اكتشاف الموجات:",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="قيمة أعلى تعني حساسية أكبر في اكتشاف الموجات"
    )
    
    indicators = st.multiselect(
        "المؤشرات الفنية:",
        options=list(get_available_indicators().keys()),
        default=["RSI", "MACD", "BB_UPPER", "BB_LOWER"],
        format_func=lambda x: get_available_indicators()[x]
    )
    
    predict_future = st.checkbox("تنبؤ بالحركة المستقبلية", value=True)
    show_sentiment = st.checkbox("تحليل المشاعر", value=True)
    show_statistics = st.checkbox("إحصاءات متقدمة", value=True)

# جلب البيانات وتحضيرها
if st.session_state.data is None or st.session_state.symbol != symbol or st.session_state.timeframe != timeframe:
    try:
        with st.spinner(f"جاري تحميل البيانات لـ {valid_symbols[symbol]}..."):
            # جلب البيانات
            raw_data = fetch_market_data(symbol, timeframe)
            
            # تحضير البيانات مع المؤشرات المطلوبة
            data = prepare_data(raw_data)
            data = add_indicators(data, indicators)
            
            # تخزين البيانات في حالة الجلسة
            st.session_state.data = data
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
            st.session_state.last_update = datetime.datetime.now()
            
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        st.stop()

# عرض البيانات في تخطيط رئيسي
data = st.session_state.data
if data is not None:
    # تقسيم الصفحة إلى ثلاثة أقسام
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.header(f"📊 تحليل {valid_symbols[symbol]}")
        
        # إنشاء تخطيط الرسم البياني
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"سعر {valid_symbols[symbol]}", "المؤشرات الفنية")
        )
        
        # إضافة شمعدان للسعر
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="السعر",
            ),
            row=1, col=1
        )
        
        # إضافة المؤشرات الفنية على الرسم البياني
        colors = {
            "SMA20": "#1f77b4",
            "SMA50": "#ff7f0e",
            "SMA200": "#2ca02c",
            "EMA10": "#d62728",
            "EMA20": "#9467bd",
            "BB_UPPER": "rgba(0, 128, 0, 0.3)",
            "BB_LOWER": "rgba(0, 128, 0, 0.3)",
            "BB_MIDDLE": "rgba(0, 128, 0, 0.7)"
        }
        
        for indicator in indicators:
            if indicator in ["SMA20", "SMA50", "SMA200", "EMA10", "EMA20"]:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            name=get_available_indicators()[indicator],
                            line=dict(color=colors.get(indicator, "blue")),
                        ),
                        row=1, col=1
                    )
            elif indicator in ["BB_UPPER", "BB_LOWER", "BB_MIDDLE"]:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            name=get_available_indicators()[indicator],
                            line=dict(color=colors.get(indicator, "green")),
                        ),
                        row=1, col=1
                    )
            elif indicator == "RSI":
                if "RSI" in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["RSI"],
                            name="مؤشر القوة النسبية",
                            line=dict(color='purple'),
                        ),
                        row=2, col=1
                    )
                    
                    # إضافة خطوط مستويات RSI
                    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            
            elif indicator == "MACD":
                if all(col in data.columns for col in ["MACD", "MACD_SIGNAL"]):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["MACD"],
                            name="MACD",
                            line=dict(color='blue'),
                        ),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["MACD_SIGNAL"],
                            name="إشارة MACD",
                            line=dict(color='red'),
                        ),
                        row=2, col=1
                    )
                    
                    # إضافة الهيستوجرام
                    if "MACD_HIST" in data.columns:
                        colors_macd_hist = ['red' if val < 0 else 'green' for val in data["MACD_HIST"]]
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data["MACD_HIST"],
                                name="MACD Histogram",
                                marker_color=colors_macd_hist,
                            ),
                            row=2, col=1
                        )
        
        # إضافة موجات إليوت إذا تم تحليلها
        if st.session_state.waves is not None:
            waves = st.session_state.waves
            wave_styles = {
                '1': {'color': 'blue', 'width': 2},
                '2': {'color': 'red', 'width': 2},
                '3': {'color': 'green', 'width': 2},
                '4': {'color': 'orange', 'width': 2},
                '5': {'color': 'purple', 'width': 2},
                'A': {'color': 'darkred', 'width': 2, 'dash': 'dash'},
                'B': {'color': 'darkorange', 'width': 2, 'dash': 'dash'},
                'C': {'color': 'darkblue', 'width': 2, 'dash': 'dash'}
            }
            
            # رسم خطوط الموجات
            for wave_type, wave_points in waves['waves'].items():
                # الاستمرار فقط إذا كانت wave_points قائمة ولها عناصر كافية
                if isinstance(wave_points, list) and len(wave_points) >= 2:
                    for i in range(len(wave_points) - 1):
                        # التحقق من أن wave_points[i] هو قاموس يحتوي على المفاتيح المطلوبة
                        if (isinstance(wave_points[i], dict) and 'idx' in wave_points[i] and 
                            isinstance(wave_points[i+1], dict) and 'idx' in wave_points[i+1]):
                            
                            start_idx = wave_points[i]['idx']
                            end_idx = wave_points[i + 1]['idx']
                            
                            if isinstance(start_idx, int) and isinstance(end_idx, int) and start_idx < len(data.index) and end_idx < len(data.index):
                                style = wave_styles.get(wave_type[0], {'color': 'gray', 'width': 1})
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[data.index[start_idx], data.index[end_idx]],
                                        y=[wave_points[i]['price'], wave_points[i + 1]['price']],
                                        mode='lines',
                                        name=f"موجة {wave_type}",
                                        line=dict(
                                            color=style['color'], 
                                            width=style['width'], 
                                            dash=style.get('dash', 'solid')
                                        ),
                                        showlegend=True
                                    ),
                                    row=1, col=1
                                )
            
            # إضافة تسميات الموجات
            for wave_type, wave_points in waves['waves'].items():
                # التحقق من نوع البيانات والمحتوى
                if isinstance(wave_points, list):
                    for point in wave_points:
                        if isinstance(point, dict) and 'idx' in point and 'price' in point:
                            if isinstance(point['idx'], int) and point['idx'] < len(data.index):
                                fig.add_annotation(
                                    x=data.index[point['idx']],
                                    y=point['price'],
                                    text=wave_type,
                                    showarrow=True,
                                    arrowhead=1,
                                    ax=0,
                                    ay=-40,
                                    row=1, col=1
                                )
        
        # تنبؤ بالحركة المستقبلية إذا تم طلبه
        if predict_future and len(data) > 30:
            with st.spinner("جاري التنبؤ بالحركة المستقبلية..."):
                # التحقق من وجود نموذج محفوظ أو تدريب نموذج جديد
                model = load_model_if_exists(symbol)
                if model is None:
                    model = train_model(data)
                
                # التنبؤ بالحركة المستقبلية
                future_days = 5
                predictions = predict_next_movement(model, data, future_days)
                
                # إضافة التنبؤات للرسم البياني
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(data.index[-30:]) + list(future_dates),
                        y=list(data['Close'][-30:]) + list(predictions),
                        name="التنبؤ",
                        line=dict(color='darkblue', width=2, dash='dot'),
                    ),
                    row=1, col=1
                )
                
                # إضافة منطقة الثقة
                confidence = 0.05 * data['Close'].mean()  # قيمة تقريبية لنطاق الثقة
                upper_bound = [p + confidence for p in predictions]
                lower_bound = [p - confidence for p in predictions]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(future_dates),
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(future_dates),
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        fillcolor='rgba(0,100,80,0.2)',
                        name="نطاق الثقة"
                    ),
                    row=1, col=1
                )
        
        # تحديث تخطيط الرسم البياني
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            title_text=f"تحليل {valid_symbols[symbol]} - {timeframe_options[timeframe]}",
            title_font_size=20,
            hovermode="x unified"
        )
        
        # تعديل نمط المحاور
        fig.update_yaxes(title_text="السعر", row=1, col=1)
        fig.update_yaxes(title_text="القيمة", row=2, col=1)
        fig.update_xaxes(title_text="التاريخ", row=2, col=1)
        
        # عرض الرسم البياني
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض أنماط الموجات المكتشفة
        if st.session_state.waves is not None:
            st.subheader("🔍 أنماط موجات إليوت المكتشفة")
            
            patterns = get_wave_patterns(st.session_state.waves)
            
            if patterns:
                pattern_cols = st.columns(3)
                for i, (pattern_name, pattern_data) in enumerate(patterns.items()):
                    with pattern_cols[i % 3]:
                        reliability = pattern_data.get('reliability', 'متوسطة')
                        color = "green" if reliability == "عالية" else "orange" if reliability == "متوسطة" else "red"
                        
                        # استخراج الاتجاه المتوقع إذا كان موجودًا
                        expected_direction = pattern_data.get('expected_direction', 'محايد')
                        direction_color = "green" if "صاعد" in expected_direction else "red" if "هابط" in expected_direction else "orange"
                        
                        st.markdown(f"""
                        <div style='border: 1px solid {color}; padding: 10px; border-radius: 5px;'>
                            <h4 style='color: {color};'>{pattern_name}</h4>
                            <p><strong>الموثوقية:</strong> {reliability}</p>
                            <p><strong>الاتجاه المتوقع:</strong> <span style='color: {direction_color};'>{expected_direction}</span></p>
                            <p><strong>الوصف:</strong> {pattern_data.get('description', 'لا يوجد وصف')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("لم يتم اكتشاف أنماط واضحة، يرجى تجربة إطار زمني آخر أو رمز مختلف.")
                
            # الأهداف المحتملة
            st.subheader("🎯 الأهداف السعرية المحتملة")
            
            current_price = data['Close'].iloc[-1]
            # تحديد أهداف افتراضية في حالة حدوث خطأ
            try:
                targets = calculate_potential_targets(st.session_state.waves, current_price)
            except Exception as e:
                st.warning(f"خطأ في حساب الأهداف: {str(e)}")
                # أهداف افتراضية
                targets = {
                    "target_1": current_price * 1.05,
                    "target_2": current_price * 1.10,
                    "target_3": current_price * 1.15,
                    "target_1_percentage": 5.0,
                    "target_2_percentage": 10.0,
                    "target_3_percentage": 15.0
                }
            
            if targets:
                target_cols = st.columns(3)
                with target_cols[0]:
                    st.metric(
                        label="الهدف الأول", 
                        value=format_number(targets['target_1']), 
                        delta=f"{format_number(targets['target_1_percentage'])}%"
                    )
                with target_cols[1]:
                    st.metric(
                        label="الهدف الثاني", 
                        value=format_number(targets['target_2']), 
                        delta=f"{format_number(targets['target_2_percentage'])}%"
                    )
                with target_cols[2]:
                    st.metric(
                        label="الهدف الثالث", 
                        value=format_number(targets['target_3']), 
                        delta=f"{format_number(targets['target_3_percentage'])}%"
                    )
            else:
                st.info("لا يمكن حساب الأهداف، يرجى تحليل الموجات أولاً.")
            
            # توصيات التداول
            st.subheader("💹 توصيات التداول")
            
            if st.session_state.waves and 'trading_signals' in st.session_state.waves:
                signals = st.session_state.waves['trading_signals']
                
                if signals:
                    signal_cols = st.columns([1, 1, 1, 1])
                    with signal_cols[0]:
                        direction_color = "green" if signals['direction'] == "شراء" else "red" if signals['direction'] == "بيع" else "gray"
                        st.markdown(f"<h3 style='color: {direction_color};'>{signals['direction']}</h3>", unsafe_allow_html=True)
                    
                    with signal_cols[1]:
                        st.metric(label="نقطة الدخول", value=format_number(signals['entry']))
                    
                    with signal_cols[2]:
                        st.metric(label="وقف الخسارة", value=format_number(signals['stop_loss']))
                    
                    with signal_cols[3]:
                        st.metric(label="جني الأرباح", value=format_number(signals['take_profit']))
                    
                    # حساب نسبة المخاطرة/المكافأة
                    risk_reward_ratio, quality = calculate_risk_reward_ratio(
                        signals['entry'], signals['stop_loss'], signals['take_profit']
                    )
                    
                    st.progress(quality / 5, f"جودة الصفقة: {quality}/5 (نسبة المخاطرة/المكافأة: {risk_reward_ratio:.2f})")
                    
                    if 'notes' in signals and signals['notes']:
                        st.info(signals['notes'])
                else:
                    st.info("لا توجد إشارات تداول حالية، قد يكون السوق في مرحلة تجميع.")
            else:
                st.info("قم بتحليل الموجات للحصول على توصيات التداول.")
            
            # اختبار الاستراتيجية
            if st.button("اختبار الاستراتيجية"):
                with st.spinner("جاري اختبار الاستراتيجية..."):
                    if st.session_state.waves and 'trading_signals' in st.session_state.waves:
                        signals = st.session_state.waves['trading_signals']
                        
                        if signals:
                            # اختبار الاستراتيجية
                            backtest_results = backtest_strategy(data, st.session_state.waves, lookback_periods=100)
                            
                            # حساب مقاييس الأداء
                            performance = calculate_performance_metrics(backtest_results)
                            
                            # عرض النتائج
                            st.subheader("📊 نتائج اختبار الاستراتيجية")
                            
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric(label="العائد الإجمالي", value=f"{performance['total_return']:.2f}%")
                            with metric_cols[1]:
                                st.metric(label="نسبة شارب", value=f"{performance['sharpe_ratio']:.2f}")
                            with metric_cols[2]:
                                st.metric(label="نسبة الفوز", value=f"{performance['win_rate']:.2f}%")
                            with metric_cols[3]:
                                st.metric(label="الحد الأقصى للسحب", value=f"{performance['max_drawdown']:.2f}%")
                            
                            # عرض منحنى الأسهم
                            fig_equity = go.Figure()
                            
                            fig_equity.add_trace(
                                go.Scatter(
                                    x=backtest_results.index,
                                    y=backtest_results['equity_curve'],
                                    name="منحنى الأسهم",
                                    line=dict(color='blue', width=2),
                                )
                            )
                            
                            fig_equity.update_layout(
                                height=400,
                                title_text="منحنى الأسهم",
                                title_font_size=16,
                                xaxis_title="التاريخ",
                                yaxis_title="رأس المال"
                            )
                            
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            # عرض الصفقات
                            st.subheader("📝 سجل الصفقات")
                            
                            if 'trades' in backtest_results:
                                trades_df = pd.DataFrame(backtest_results['trades'])
                                if not trades_df.empty:
                                    st.dataframe(trades_df, use_container_width=True)
                                else:
                                    st.info("لم يتم تنفيذ أي صفقات خلال فترة الاختبار.")
                            else:
                                st.info("لم يتم تنفيذ أي صفقات خلال فترة الاختبار.")
                        else:
                            st.warning("لا توجد إشارات تداول لاختبارها.")
                    else:
                        st.warning("قم بتحليل الموجات أولاً للحصول على إشارات التداول.")
    
    # القسم الجانبي للمعلومات الإضافية
    with col2:
        # معلومات عن الرمز المالي
        st.header("ℹ️ معلومات الرمز")
        
        price_info_cols = st.columns(2)
        with price_info_cols[0]:
            current_price = format_number(data['Close'].iloc[-1])
            previous_price = format_number(data['Close'].iloc[-2])
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            change_color = "green" if price_change >= 0 else "red"
            change_icon = "↗" if price_change >= 0 else "↘"
            
            st.metric(
                label="السعر الحالي",
                value=current_price,
                delta=f"{change_icon} {format_number(price_change_pct)}%"
            )
        
        with price_info_cols[1]:
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_change = (volume / avg_volume - 1) * 100
            
            st.metric(
                label="الحجم",
                value=format_number(volume),
                delta=f"{format_number(volume_change)}% من المتوسط"
            )
        
        # مؤشرات فنية إضافية
        st.subheader("🔍 المؤشرات الفنية")
        
        # RSI
        if "RSI" in data.columns:
            rsi_value = data['RSI'].iloc[-1]
            rsi_status = (
                "ذروة شراء" if rsi_value > 70 else
                "ذروة بيع" if rsi_value < 30 else
                "محايد"
            )
            rsi_color = (
                "red" if rsi_value > 70 else
                "green" if rsi_value < 30 else
                "orange"
            )
            
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span>مؤشر القوة النسبية (RSI)</span>
                <span style='color: {rsi_color};'>{rsi_value:.2f} ({rsi_status})</span>
            </div>
            """, unsafe_allow_html=True)
        
        # MACD
        if all(col in data.columns for col in ["MACD", "MACD_SIGNAL"]):
            macd_value = data['MACD'].iloc[-1]
            signal_value = data['MACD_SIGNAL'].iloc[-1]
            macd_diff = macd_value - signal_value
            
            macd_status = "إشارة شراء" if macd_diff > 0 else "إشارة بيع"
            macd_color = "green" if macd_diff > 0 else "red"
            
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span>MACD</span>
                <span style='color: {macd_color};'>{macd_value:.2f} ({macd_status})</span>
            </div>
            """, unsafe_allow_html=True)
        
        # المتوسطات المتحركة
        for ma in ["SMA20", "SMA50", "SMA200"]:
            if ma in data.columns:
                ma_value = data[ma].iloc[-1]
                price = data['Close'].iloc[-1]
                
                ma_diff = ((price / ma_value) - 1) * 100
                ma_status = "فوق المتوسط" if price > ma_value else "تحت المتوسط"
                ma_color = "green" if price > ma_value else "red"
                
                ma_name = ma.replace("SMA", "المتوسط المتحرك البسيط ")
                
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span>{ma_name}</span>
                    <span style='color: {ma_color};'>{format_number(ma_value)} ({ma_status}, {format_number(ma_diff)}%)</span>
                </div>
                """, unsafe_allow_html=True)
        
        # تحليل المشاعر إذا تم طلبه
        if show_sentiment:
            st.subheader("🔮 تحليل المشاعر")
            
            with st.spinner("جاري تحليل المشاعر..."):
                try:
                    # تحليل مشاعر السوق
                    market_sentiment = get_market_sentiment(symbol)
                    
                    # تحليل كبار المتداولين
                    top_traders_sentiment = top_traders_analyzer.get_top_traders_sentiment(symbol)
                    
                    # تحليل اتجاهات السوق
                    market_trend = get_market_trends(symbol)
                    
                    # عرض مشاعر السوق
                    sentiment_score = market_sentiment['sentiment_score']
                    sentiment_color = (
                        "green" if sentiment_score > 0.1 else
                        "red" if sentiment_score < -0.1 else
                        "orange"
                    )
                    
                    st.markdown(f"""
                    <div style='padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid {sentiment_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span><strong>المشاعر العامة:</strong></span>
                            <span style='color: {sentiment_color};'>{market_sentiment['overall_sentiment']}</span>
                        </div>
                        <div style='margin-top: 5px;'>
                            <small>قوة المشاعر: {market_sentiment['sentiment_strength']:.1f}%</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # عرض توزيع كبار المتداولين
                    st.markdown("<strong>توزيع كبار المتداولين:</strong>", unsafe_allow_html=True)
                    
                    # تحليل موقف المؤسسات
                    institutional_positioning = top_traders_analyzer.get_institutional_positioning(symbol)
                    
                    # عرض نسب المشترين والبائعين (استخدم قيم افتراضية إذا كانت المفاتيح غير موجودة)
                    buyers = institutional_positioning.get('buyers_percentage', 50)
                    sellers = institutional_positioning.get('sellers_percentage', 50)
                    
                    st.progress(buyers/100, f"المشترين: {buyers:.1f}% | البائعين: {sellers:.1f}%")
                    
                    # اتجاه المال الذكي
                    smart_money_direction = institutional_positioning.get('smart_money_direction', 'محايد')
                    smart_money_color = "green" if smart_money_direction == "شراء" else "red" if smart_money_direction == "بيع" else "gray"
                    
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span>اتجاه المال الذكي:</span>
                        <span style='color: {smart_money_color};'>{smart_money_direction}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # عرض اتجاهات السوق
                    st.markdown("<strong>اتجاهات السوق:</strong>", unsafe_allow_html=True)
                    
                    # التحقق من وجود المفاتيح قبل استخدامها
                    trend = market_trend.get('overall_trend', 'محايد')
                    volatility = market_trend.get('volatility', 'متوسطة')
                    risk_level = market_trend.get('strength', {}).get('short_term', 50)
                    
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span>الاتجاه:</span>
                        <span>{trend}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span>التقلب:</span>
                        <span>{volatility}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span>مستوى المخاطرة:</span>
                        <span>{risk_level}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # عرض العناوين الرئيسية
                    if 'top_headlines' in market_sentiment and market_sentiment['top_headlines']:
                        with st.expander("أهم الأخبار"):
                            for headline in market_sentiment['top_headlines'][:3]:
                                st.markdown(f"- {headline}")
                
                    # إضافة قسم جديد لإشارات التداول عالية الاحتمالية
                    st.subheader("🎯 إشارة تداول عالية الاحتمالية (90%+)")
                    
                    if st.session_state.waves is not None:
                        with st.spinner("جاري تحليل فرص التداول عالية الاحتمالية..."):
                            # توليد إشارة التداول عالية الاحتمالية
                            trade_signal = generate_high_probability_signal(
                                data=data, 
                                elliott_waves=st.session_state.waves, 
                                technical_data=market_trend,
                                sentiment_data=market_sentiment,
                                confidence_threshold=0.95  # تحسين مستوى الثقة إلى 95% لضمان دقة أعلى
                            )
                            
                            # التحقق من صلاحية الإشارة
                            is_valid, validation_message = validate_trading_opportunity(
                                data=data,
                                signal_data=trade_signal,
                                min_risk_reward=3.0,  # زيادة نسبة المخاطرة/المكافأة للصفقات عالية الجودة
                                min_confidence=0.95  # تحسين مستوى الثقة ليتماشى مع معيار 95%
                            )
                            
                            # عرض إشارة التداول
                            signal_type = trade_signal["signal"]
                            confidence = trade_signal["confidence"]
                            
                            if signal_type != "محايد" and confidence >= 95:  # رفع الحد الأدنى لمستوى الثقة المطلوب
                                # تنسيق إشارة التداول
                                signal_text = format_trade_signal(trade_signal)
                                
                                # إنشاء مربع بلون مناسب للإشارة
                                signal_color = "rgba(0, 128, 0, 0.2)" if signal_type == "شراء" else "rgba(255, 0, 0, 0.2)"
                                
                                st.markdown(f"""
                                <div style='background-color: {signal_color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
                                    {signal_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("لا توجد إشارة تداول عالية الاحتمالية في الوقت الحالي. راقب السوق للحصول على فرص أفضل.")
                    else:
                        st.warning("يرجى تحليل موجات إليوت أولاً للحصول على إشارات تداول عالية الاحتمالية")
                
                except Exception as e:
                    st.error(f"خطأ في تحليل المشاعر: {str(e)}")
        
        # إحصاءات متقدمة
        if show_statistics:
            st.subheader("📊 إحصاءات متقدمة")
            
            try:
                # حساب التقلب
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5) * 100  # التقلب السنوي
                
                # حساب المدى اليومي
                daily_range = ((data['High'] - data['Low']) / data['Close']).mean() * 100
                
                # حساب أعلى وأدنى سعر في الفترة المعروضة
                high = data['High'].max()
                low = data['Low'].min()
                
                # عرض الإحصاءات
                stat_cols = st.columns(2)
                
                with stat_cols[0]:
                    st.metric(label="التقلب", value=f"{volatility:.2f}%")
                    st.metric(label="أعلى سعر", value=format_number(high))
                
                with stat_cols[1]:
                    st.metric(label="المدى اليومي", value=f"{daily_range:.2f}%")
                    st.metric(label="أدنى سعر", value=format_number(low))
                
                # نطاقات دعم ومقاومة
                st.markdown("<strong>مستويات الدعم والمقاومة:</strong>", unsafe_allow_html=True)
                
                last_close = data['Close'].iloc[-1]
                
                # حساب مستويات بسيطة
                resistance_levels = [
                    last_close * 1.01,  # +1%
                    last_close * 1.02,  # +2%
                    last_close * 1.05   # +5%
                ]
                
                support_levels = [
                    last_close * 0.99,  # -1%
                    last_close * 0.98,  # -2%
                    last_close * 0.95   # -5%
                ]
                
                # عرض مستويات المقاومة
                st.markdown("**مستويات المقاومة:**")
                for i, level in enumerate(reversed(resistance_levels)):
                    st.markdown(f"{i+1}. {format_number(level)}")
                
                # عرض مستويات الدعم
                st.markdown("**مستويات الدعم:**")
                for i, level in enumerate(support_levels):
                    st.markdown(f"{i+1}. {format_number(level)}")
            
            except Exception as e:
                st.error(f"خطأ في حساب الإحصاءات: {str(e)}")

# الجزء السفلي - معلومات إضافية
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>محلل موجات إليوت - نسخة 1.0</p>
    <p>تحليل الأسواق المالية باستخدام نظرية موجات إليوت مع دمج الذكاء الاصطناعي</p>
    <p>هذا التطبيق مخصص للأغراض التعليمية فقط وليس توصية استثمارية.</p>
</div>
""", unsafe_allow_html=True)
