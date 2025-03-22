import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import datetime

def backtest_strategy(data: pd.DataFrame, waves_data: Dict, lookback_periods: int = 100) -> pd.DataFrame:
    """
    اختبار استراتيجية التداول باستخدام تحليل موجات إليوت
    
    المعلمات:
    ----------
    data : pd.DataFrame
        إطار البيانات مع بيانات الأسعار
    waves_data : Dict
        قاموس يحتوي على معلومات عن الموجات المكتشفة وإشارات التداول
    lookback_periods : int
        عدد الفترات السابقة التي سيتم اختبارها
        
    العائدات:
    -------
    pd.DataFrame
        نتائج الاختبار البرمجي
    """
    if data is None or len(data) < lookback_periods:
        return pd.DataFrame()
    
    # استخدام الفترات السابقة من البيانات للاختبار
    backtest_data = data.iloc[-lookback_periods:].copy()
    
    # التأكد من وجود إشارات تداول
    if 'trading_signals' not in waves_data or not waves_data['trading_signals']:
        return pd.DataFrame()
    
    trading_signals = waves_data['trading_signals']
    
    # إعداد متغيرات الاختبار
    initial_capital = 10000.0  # رأس المال الافتراضي
    position_size = 0.1  # حجم المركز (10% من رأس المال)
    commission = 0.001  # عمولة التداول (0.1%)
    
    # إنشاء أعمدة لتتبع التداول
    backtest_data['position'] = 0  # 1 للشراء، -1 للبيع، 0 للحياد
    backtest_data['entry_price'] = np.nan
    backtest_data['exit_price'] = np.nan
    backtest_data['stop_loss'] = np.nan
    backtest_data['take_profit'] = np.nan
    backtest_data['trade_result'] = 0.0
    backtest_data['capital'] = initial_capital
    backtest_data['equity_curve'] = initial_capital
    
    # بدء الاختبار
    current_position = 0
    entry_price = 0.0
    entry_date = None
    stop_loss = 0.0
    take_profit = 0.0
    current_capital = initial_capital
    trades = []
    
    # تحديد اتجاه الإشارة
    direction = trading_signals['direction']
    signal_type = 1 if 'شراء' in direction else -1 if 'بيع' in direction else 0
    
    # تطبيق الإشارة على الفترة الزمنية
    for i in range(1, len(backtest_data)):
        current_price = backtest_data['Close'].iloc[i]
        previous_price = backtest_data['Close'].iloc[i-1]
        current_date = backtest_data.index[i]
        
        # تحديث قيم وقف الخسارة وجني الأرباح
        backtest_data.loc[backtest_data.index[i], 'stop_loss'] = trading_signals['stop_loss']
        backtest_data.loc[backtest_data.index[i], 'take_profit'] = trading_signals['take_profit']
        
        # فتح مركز جديد إذا لم يكن هناك مركز مفتوح وتطابقت الشروط
        if current_position == 0:
            # افتراض فتح مركز جديد في أول يوم من الاختبار (للتبسيط)
            if i == 1:
                current_position = signal_type
                entry_price = current_price
                entry_date = current_date
                stop_loss = trading_signals['stop_loss']
                take_profit = trading_signals['take_profit']
                
                # تسجيل الدخول
                backtest_data.loc[backtest_data.index[i], 'position'] = current_position
                backtest_data.loc[backtest_data.index[i], 'entry_price'] = entry_price
        
        # إغلاق المركز الحالي بناءً على وقف الخسارة أو جني الأرباح
        elif current_position != 0:
            exit_triggered = False
            exit_type = ""
            exit_price = current_price
            
            # التحقق من تحقق وقف الخسارة
            if (current_position == 1 and current_price <= stop_loss) or \
               (current_position == -1 and current_price >= stop_loss):
                exit_triggered = True
                exit_type = "وقف الخسارة"
                exit_price = stop_loss  # تقريب لسعر التنفيذ عند وقف الخسارة
            
            # التحقق من تحقق جني الأرباح
            elif (current_position == 1 and current_price >= take_profit) or \
                 (current_position == -1 and current_price <= take_profit):
                exit_triggered = True
                exit_type = "جني الأرباح"
                exit_price = take_profit  # تقريب لسعر التنفيذ عند جني الأرباح
            
            # إغلاق المركز في نهاية فترة الاختبار
            elif i == len(backtest_data) - 1:
                exit_triggered = True
                exit_type = "إغلاق نهاية الفترة"
            
            # إذا تم تحقق شرط الخروج
            if exit_triggered:
                # حساب نتيجة الصفقة
                position_value = current_capital * position_size
                num_units = position_value / entry_price
                
                if current_position == 1:  # شراء
                    profit_loss = num_units * (exit_price - entry_price)
                    profit_loss_pct = (exit_price / entry_price - 1) * 100
                else:  # بيع
                    profit_loss = num_units * (entry_price - exit_price)
                    profit_loss_pct = (entry_price / exit_price - 1) * 100
                
                # خصم العمولة
                profit_loss = profit_loss - (position_value * commission * 2)  # عمولة للدخول والخروج
                
                # تحديث رأس المال
                current_capital += profit_loss
                
                # تسجيل نتيجة الصفقة
                backtest_data.loc[backtest_data.index[i], 'exit_price'] = exit_price
                backtest_data.loc[backtest_data.index[i], 'trade_result'] = profit_loss
                
                # إضافة الصفقة إلى سجل الصفقات
                trade_info = {
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'direction': 'شراء' if current_position == 1 else 'بيع',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'trade_duration': (current_date - entry_date).days
                }
                trades.append(trade_info)
                
                # إعادة تعيين المتغيرات
                current_position = 0
                entry_price = 0.0
                entry_date = None
            
        # تحديث قيمة رأس المال ومنحنى الأسهم
        backtest_data.loc[backtest_data.index[i], 'capital'] = current_capital
        
        # تحديث منحنى الأسهم (رأس المال + قيمة المراكز المفتوحة)
        if current_position != 0:
            position_value = current_capital * position_size
            num_units = position_value / entry_price
            
            if current_position == 1:  # شراء
                unrealized_pnl = num_units * (current_price - entry_price)
            else:  # بيع
                unrealized_pnl = num_units * (entry_price - current_price)
                
            equity = current_capital + unrealized_pnl
        else:
            equity = current_capital
            
        backtest_data.loc[backtest_data.index[i], 'equity_curve'] = equity
    
    # إضافة سجل الصفقات
    backtest_data = backtest_data.copy()
    backtest_data.attrs['trades'] = trades
    
    return backtest_data

def calculate_performance_metrics(backtest_results: pd.DataFrame) -> Dict:
    """
    حساب مقاييس أداء الاستراتيجية
    
    المعلمات:
    ----------
    backtest_results : pd.DataFrame
        نتائج الاختبار البرمجي
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على مقاييس الأداء
    """
    if backtest_results.empty:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'num_trades': 0
        }
    
    # استخراج منحنى الأسهم
    equity_curve = backtest_results['equity_curve']
    
    # استخراج سجل الصفقات
    trades = getattr(backtest_results, 'attrs', {}).get('trades', [])
    
    # حساب العائد الإجمالي
    initial_capital = equity_curve.iloc[0]
    final_capital = equity_curve.iloc[-1]
    total_return = ((final_capital / initial_capital) - 1) * 100
    
    # حساب العائد السنوي
    trading_days = len(backtest_results)
    trading_years = trading_days / 252  # افتراض 252 يوم تداول في السنة
    annualized_return = ((1 + total_return / 100) ** (1 / trading_years) - 1) * 100 if trading_years > 0 else 0
    
    # حساب الانحراف المعياري
    daily_returns = equity_curve.pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100  # الانحراف المعياري السنوي
    
    # حساب نسبة شارب
    risk_free_rate = 0.02  # معدل خالي من المخاطر (2%)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # حساب أقصى انخفاض
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max - 1) * 100
    max_drawdown = abs(drawdown.min())
    
    # حساب إحصاءات الصفقات
    if trades:
        # عدد الصفقات
        num_trades = len(trades)
        
        # عدد الصفقات الرابحة
        winning_trades = [trade for trade in trades if trade['profit_loss'] > 0]
        num_winning_trades = len(winning_trades)
        
        # نسبة الفوز
        win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
        
        # متوسط الصفقة
        avg_trade = sum(trade['profit_loss'] for trade in trades) / num_trades if num_trades > 0 else 0
        
        # عامل الربح
        total_profit = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0)
        total_loss = abs(sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # متوسط الربح ومتوسط الخسارة
        avg_profit = sum(trade['profit_loss'] for trade in winning_trades) / num_winning_trades if num_winning_trades > 0 else 0
        avg_loss = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] < 0) / (num_trades - num_winning_trades) if (num_trades - num_winning_trades) > 0 else 0
        
        # متوسط مدة الصفقة
        avg_duration = sum(trade['trade_duration'] for trade in trades) / num_trades if num_trades > 0 else 0
    else:
        # قيم افتراضية في حالة عدم وجود صفقات
        num_trades = 0
        win_rate = 0
        avg_trade = 0
        profit_factor = 0
        avg_profit = 0
        avg_loss = 0
        avg_duration = 0
    
    # جمع كل المقاييس في قاموس
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'num_trades': num_trades,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'avg_duration': avg_duration
    }
    
    return metrics

def calculate_strategy_exposure(backtest_results: pd.DataFrame) -> Dict:
    """
    حساب تعرض الاستراتيجية للسوق
    
    المعلمات:
    ----------
    backtest_results : pd.DataFrame
        نتائج الاختبار البرمجي
        
    العائدات:
    -------
    Dict
        قاموس يحتوي على إحصاءات التعرض
    """
    if backtest_results.empty:
        return {
            'long_exposure': 0,
            'short_exposure': 0,
            'total_exposure': 0,
            'average_exposure': 0
        }
    
    # استخراج مواقف التداول
    positions = backtest_results['position']
    
    # حساب إحصاءات التعرض
    total_periods = len(positions)
    long_periods = (positions > 0).sum()
    short_periods = (positions < 0).sum()
    neutral_periods = (positions == 0).sum()
    
    # حساب نسب التعرض
    long_exposure = (long_periods / total_periods) * 100 if total_periods > 0 else 0
    short_exposure = (short_periods / total_periods) * 100 if total_periods > 0 else 0
    total_exposure = ((long_periods + short_periods) / total_periods) * 100 if total_periods > 0 else 0
    average_exposure = (long_exposure - short_exposure) if total_periods > 0 else 0
    
    return {
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'total_exposure': total_exposure,
        'average_exposure': average_exposure,
        'neutral_periods': neutral_periods
    }

def analyze_drawdowns(backtest_results: pd.DataFrame, num_drawdowns: int = 5) -> List[Dict]:
    """
    تحليل فترات الانخفاض الرئيسية
    
    المعلمات:
    ----------
    backtest_results : pd.DataFrame
        نتائج الاختبار البرمجي
    num_drawdowns : int
        عدد فترات الانخفاض المراد تحليلها
        
    العائدات:
    -------
    List[Dict]
        قائمة تحتوي على معلومات عن فترات الانخفاض
    """
    if backtest_results.empty:
        return []
    
    # استخراج منحنى الأسهم
    equity_curve = backtest_results['equity_curve']
    
    # حساب الذروة المتراكمة
    rolling_max = equity_curve.cummax()
    
    # حساب الانخفاض
    drawdown = (equity_curve / rolling_max - 1) * 100
    
    # تحديد بداية ونهاية فترات الانخفاض
    drawdown_started = False
    drawdown_periods = []
    current_drawdown = {}
    
    for i in range(len(drawdown)):
        if not drawdown_started and drawdown.iloc[i] < 0:
            # بداية فترة انخفاض جديدة
            drawdown_started = True
            current_drawdown = {
                'start_date': drawdown.index[i],
                'start_equity': equity_curve.iloc[i],
                'max_drawdown': drawdown.iloc[i],
                'max_drawdown_date': drawdown.index[i]
            }
        
        elif drawdown_started:
            # تحديث أقصى انخفاض
            if drawdown.iloc[i] < current_drawdown['max_drawdown']:
                current_drawdown['max_drawdown'] = drawdown.iloc[i]
                current_drawdown['max_drawdown_date'] = drawdown.index[i]
            
            # التحقق من انتهاء فترة الانخفاض
            if drawdown.iloc[i] == 0:
                drawdown_started = False
                current_drawdown['end_date'] = drawdown.index[i]
                current_drawdown['end_equity'] = equity_curve.iloc[i]
                current_drawdown['recovery_time'] = (current_drawdown['end_date'] - current_drawdown['max_drawdown_date']).days
                current_drawdown['drawdown_time'] = (current_drawdown['max_drawdown_date'] - current_drawdown['start_date']).days
                current_drawdown['total_time'] = (current_drawdown['end_date'] - current_drawdown['start_date']).days
                
                drawdown_periods.append(current_drawdown)
    
    # إذا كانت آخر فترة انخفاض لا تزال مستمرة
    if drawdown_started:
        current_drawdown['end_date'] = drawdown.index[-1]
        current_drawdown['end_equity'] = equity_curve.iloc[-1]
        current_drawdown['recovery_time'] = 0  # لم يتم التعافي بعد
        current_drawdown['drawdown_time'] = (current_drawdown['max_drawdown_date'] - current_drawdown['start_date']).days
        current_drawdown['total_time'] = (current_drawdown['end_date'] - current_drawdown['start_date']).days
        
        drawdown_periods.append(current_drawdown)
    
    # ترتيب فترات الانخفاض حسب الحجم
    drawdown_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])
    
    # إرجاع أكبر عدد محدد من فترات الانخفاض
    return drawdown_periods[:num_drawdowns]

def generate_monte_carlo_simulation(backtest_results: pd.DataFrame, num_simulations: int = 100, confidence_level: float = 0.95) -> Dict:
    """
    إنشاء محاكاة مونتي كارلو لاختبار متانة الاستراتيجية
    
    المعلمات:
    ----------
    backtest_results : pd.DataFrame
        نتائج الاختبار البرمجي
    num_simulations : int
        عدد المحاكاة
    confidence_level : float
        مستوى الثقة
        
    العائدات:
    -------
    Dict
        نتائج المحاكاة
    """
    if backtest_results.empty:
        return {
            'simulated_returns': [],
            'expected_return': 0.0,
            'worst_return': 0.0,
            'best_return': 0.0,
            'confidence_interval': (0.0, 0.0)
        }
    
    # استخراج سجل الصفقات
    trades = getattr(backtest_results, 'attrs', {}).get('trades', [])
    
    if not trades:
        return {
            'simulated_returns': [],
            'expected_return': 0.0,
            'worst_return': 0.0,
            'best_return': 0.0,
            'confidence_interval': (0.0, 0.0)
        }
    
    # تجميع نتائج الصفقات
    trade_results = [trade['profit_loss_pct'] for trade in trades]
    
    # إنشاء محاكاة مونتي كارلو
    simulated_returns = []
    initial_capital = 10000.0
    
    for _ in range(num_simulations):
        # اختيار عينات عشوائية من نتائج الصفقات
        simulated_trade_results = np.random.choice(trade_results, size=len(trade_results), replace=True)
        
        # حساب العائد التراكمي
        capital = initial_capital
        for result in simulated_trade_results:
            capital *= (1 + result / 100)
        
        # حساب العائد الإجمالي
        total_return = ((capital / initial_capital) - 1) * 100
        simulated_returns.append(total_return)
    
    # ترتيب العوائد
    simulated_returns.sort()
    
    # حساب الإحصاءات
    expected_return = np.mean(simulated_returns)
    worst_return = simulated_returns[0]
    best_return = simulated_returns[-1]
    
    # حساب فترة الثقة
    lower_idx = int((1 - confidence_level) / 2 * num_simulations)
    upper_idx = int((1 - (1 - confidence_level) / 2) * num_simulations)
    confidence_interval = (simulated_returns[lower_idx], simulated_returns[upper_idx])
    
    return {
        'simulated_returns': simulated_returns,
        'expected_return': expected_return,
        'worst_return': worst_return,
        'best_return': best_return,
        'confidence_interval': confidence_interval
    }
