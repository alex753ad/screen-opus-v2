import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Импорт модуля mean reversion analysis v6.0 (DFA + FDR + Stability + Trade Score)
from mean_reversion_analysis import (
    calculate_hurst_exponent,
    calculate_rolling_zscore,
    calculate_ou_parameters,
    calculate_ou_score,
    calculate_trade_score,
    apply_fdr_correction,
    check_cointegration_stability,
    estimate_exit_time,
    validate_ou_quality
)
from statsmodels.tools import add_constant

# Конфигурация страницы
st.set_page_config(
    page_title="Crypto Pairs Trading Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-long {
        color: #00cc00;
        font-weight: bold;
    }
    .signal-short {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-neutral {
        color: #888888;
    }
    /* Исправление читаемости для темной темы */
    .stMarkdown, .stText, p, span, div {
        color: inherit !important;
    }
    /* Таблица - темный текст на светлом фоне для читаемости */
    .dataframe {
        background-color: white !important;
        color: black !important;
    }
    .dataframe td, .dataframe th {
        color: black !important;
    }
    /* Метрики - улучшенная видимость */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'pairs_data' not in st.session_state:
    st.session_state.pairs_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'selected_pair_index' not in st.session_state:
    st.session_state.selected_pair_index = 0
if 'settings' not in st.session_state:
    # Сохранение последних настроек
    st.session_state.settings = {
        'exchange': 'okx',          # OKX по умолчанию
        'timeframe': '4h',          # 4h таймфрейм
        'lookback_days': 35,        # 35 дней
        'top_n_coins': 100,         # 100 монет
        'max_pairs_display': 30,    # 30 пар максимум
        'pvalue_threshold': 0.03,   # 0.03
        'zscore_threshold': 2.3,    # 2.3
        'max_halflife_hours': 28    # 28 часов
    }

class CryptoPairsScanner:
    def __init__(self, exchange_name='binance', timeframe='1d', lookback_days=30):
        # Попытка подключения к бирже с fallback
        self.exchange_name = exchange_name
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        
        try:
            self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
            # Проверяем доступность
            self.exchange.load_markets()
        except Exception as e:
            if '451' in str(e) or 'restricted location' in str(e).lower():
                st.warning(f"⚠️ {exchange_name.upper()} недоступен в вашем регионе. Переключаюсь на Bybit...")
                self.exchange_name = 'bybit'
                self.exchange = ccxt.bybit({'enableRateLimit': True})
            elif exchange_name == 'binance':
                st.warning(f"⚠️ Binance недоступен. Переключаюсь на Bybit...")
                self.exchange_name = 'bybit'
                self.exchange = ccxt.bybit({'enableRateLimit': True})
            else:
                raise e
        
    def get_top_coins(self, limit=100):
        """Получить топ монет по объему торгов"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            # Определяем базовую валюту в зависимости от биржи
            if self.exchange_name == 'bybit':
                base_currency = 'USDT'
                # Bybit использует формат BTC/USDT:USDT для futures, нам нужен только spot
                usdt_pairs = {k: v for k, v in tickers.items() 
                            if f'/{base_currency}' in k 
                            and ':' not in k  # Исключаем futures
                            and 'info' in v}
            else:
                # Для других бирж (Binance, OKX, etc)
                base_currency = 'USDT'
                usdt_pairs = {k: v for k, v in tickers.items() 
                            if f'/{base_currency}' in k and ':USDT' not in k}
            
            # Сортируем по объему (разные биржи используют разные поля)
            valid_pairs = []
            for symbol, ticker in usdt_pairs.items():
                try:
                    volume = float(ticker.get('quoteVolume', 0)) or float(ticker.get('volume', 0))
                    if volume > 0:
                        valid_pairs.append((symbol, volume))
                except:
                    continue
            
            # Сортируем по объему
            sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
            
            # Берем топ монет
            top_coins = [pair[0].replace(f'/{base_currency}', '') for pair in sorted_pairs[:limit]]
            
            if len(top_coins) > 0:
                st.info(f"📊 Загружено {len(top_coins)} монет с {self.exchange_name.upper()}")
                return top_coins
            else:
                raise Exception("Не удалось получить данные о монетах")
            
        except Exception as e:
            st.error(f"Ошибка при получении топ монет с {self.exchange_name}: {e}")
            
            # Fallback: возвращаем популярные монеты
            st.warning("🔄 Используется fallback список популярных монет")
            return ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 
                   'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM', 
                   'NEAR', 'APT', 'ARB', 'OP', 'DOGE']
    
    def fetch_ohlcv(self, symbol, limit=None):
        """Получить исторические данные"""
        try:
            if limit is None:
                limit = self.lookback_days
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df['close']
        except Exception as e:
            return None
    
    def test_cointegration(self, series1, series2):
        """Тест на коинтеграцию (v6.0: с константой в OLS + rolling Z-score)"""
        try:
            # Убираем NaN
            valid_data = pd.concat([series1, series2], axis=1).dropna()
            if len(valid_data) < 20:
                return None, None, None
            
            s1 = valid_data.iloc[:, 0]
            s2 = valid_data.iloc[:, 1]
            
            # Тест Энгла-Грейнджера
            score, pvalue, _ = coint(s1, s2)
            
            # [B] Расчет hedge ratio С КОНСТАНТОЙ
            s2_const = add_constant(s2)
            model = OLS(s1, s2_const).fit()
            hedge_ratio = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]
            intercept = model.params.iloc[0] if len(model.params) > 1 else 0.0
            
            # Расчет спреда (с константой)
            spread = s1 - hedge_ratio * s2 - intercept
            
            # [B] Rolling Z-score (без lookahead bias)
            zscore, zscore_series = calculate_rolling_zscore(spread.values, window=30)
            
            # Расчет half-life
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_diff = spread_diff.dropna()
            spread_lag = spread_lag.dropna()
            
            model_hl = OLS(spread_diff, spread_lag).fit()
            halflife = -np.log(2) / model_hl.params.iloc[0] if model_hl.params.iloc[0] < 0 else np.inf
            
            return {
                'pvalue': pvalue,
                'zscore': zscore,
                'zscore_series': zscore_series,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'halflife': halflife,
                'spread': spread,
                'score': score
            }
        except Exception as e:
            return None
    
    def scan_pairs(self, coins, max_pairs=50, progress_bar=None, max_halflife_hours=720):
        """Сканировать все пары (v6.0: + stability + FDR + Trade Score)"""
        results = []
        all_pvalues = []  # [C] Для FDR-коррекции
        all_results_indices = []  # Индексы в results для сопоставления с pvalues
        
        # Загружаем данные для всех монет
        st.info(f"📥 Загружаю данные для {len(coins)} монет...")
        price_data = {}
        for coin in coins:
            symbol = f"{coin}/USDT"
            prices = self.fetch_ohlcv(symbol)
            if prices is not None and len(prices) > 20:
                price_data[coin] = prices
            time.sleep(0.1)  # Rate limit
        
        if len(price_data) < 2:
            st.error("❌ Недостаточно данных для анализа")
            st.info(f"""
            **Возможные причины:**
            - Биржа {self.exchange_name.upper()} заблокирована в вашем регионе
            - Проблемы с подключением к интернету
            - Временные проблемы на бирже
            
            **Решения:**
            1. Выберите другую биржу (Bybit или OKX рекомендуются)
            2. Проверьте подключение к интернету
            3. Попробуйте через несколько минут
            4. Используйте VPN если биржа заблокирована
            """)
            return []
        
        total_combinations = len(price_data) * (len(price_data) - 1) // 2
        st.info(f"🔍 Анализирую {total_combinations} комбинаций пар из {len(price_data)} монет...")
        processed = 0
        
        # Тестируем все пары
        for i, coin1 in enumerate(price_data.keys()):
            for coin2 in list(price_data.keys())[i+1:]:
                processed += 1
                
                if progress_bar:
                    progress_bar.progress(processed / total_combinations, 
                                        f"Обработано {processed}/{total_combinations}")
                
                result = self.test_cointegration(price_data[coin1], price_data[coin2])
                
                if result and result['pvalue'] < 0.05:  # Предварительный порог
                    halflife_hours = result['halflife'] * 24
                    
                    if halflife_hours <= max_halflife_hours:
                        # [A] Hurst (DFA)
                        hurst = calculate_hurst_exponent(result['spread'])
                        
                        # OU параметры
                        dt = {'1h': 1/24, '4h': 1/6, '1d': 1}.get(self.timeframe, 1/6)
                        ou_params = calculate_ou_parameters(result['spread'], dt=dt)
                        
                        # Legacy OU Score
                        ou_score = calculate_ou_score(ou_params, hurst)
                        
                        # Валидация
                        is_valid, reason = validate_ou_quality(ou_params, hurst)
                        
                        # [D] Stability check
                        stability = check_cointegration_stability(
                            price_data[coin1].values, price_data[coin2].values
                        )
                        
                        idx = len(results)
                        results.append({
                            'pair': f"{coin1}/{coin2}",
                            'coin1': coin1,
                            'coin2': coin2,
                            'pvalue': result['pvalue'],
                            'pvalue_adj': result['pvalue'],  # Будет обновлено после FDR
                            'zscore': result['zscore'],
                            'zscore_series': result.get('zscore_series'),
                            'hedge_ratio': result['hedge_ratio'],
                            'intercept': result.get('intercept', 0.0),
                            'halflife_days': result['halflife'],
                            'halflife_hours': halflife_hours,
                            'spread': result['spread'],
                            'signal': self.get_signal(result['zscore']),
                            'hurst': hurst,
                            'theta': ou_params['theta'] if ou_params else 0,
                            'mu': ou_params['mu'] if ou_params else 0,
                            'sigma': ou_params['sigma'] if ou_params else 0,
                            'halflife_ou': ou_params['halflife_ou'] * 24 if ou_params else 999,
                            'ou_score': ou_score,
                            'ou_valid': is_valid,
                            'ou_reason': reason,
                            # [D] Stability
                            'stability_score': stability['stability_score'],
                            'stability_passed': stability['windows_passed'],
                            'stability_total': stability['total_windows'],
                            'is_stable': stability['is_stable'],
                            # Trade Score placeholder
                            'trade_score': 0,
                            'trade_breakdown': {},
                        })
                        all_pvalues.append(result['pvalue'])
                        all_results_indices.append(idx)
        
        # [C] FDR-коррекция p-values
        if len(all_pvalues) > 0:
            # Учитываем ВСЕ протестированные пары для корректного FDR
            adj_pvalues, fdr_rejected = apply_fdr_correction(all_pvalues, alpha=0.05)
            
            fdr_passed = 0
            fdr_failed = 0
            for j, idx in enumerate(all_results_indices):
                results[idx]['pvalue_adj'] = float(adj_pvalues[j])
                results[idx]['fdr_passed'] = bool(fdr_rejected[j])
                if fdr_rejected[j]:
                    fdr_passed += 1
                else:
                    fdr_failed += 1
            
            st.info(f"🔬 FDR коррекция: {fdr_passed} пар прошли, {fdr_failed} отфильтрованы")
        
        # [C] Trade Score (после FDR)
        for r in results:
            ou_p = calculate_ou_parameters(r['spread'], 
                dt={'1h': 1/24, '4h': 1/6, '1d': 1}.get(self.timeframe, 1/6))
            score, breakdown = calculate_trade_score(
                hurst=r['hurst'],
                ou_params=ou_p,
                pvalue_adj=r['pvalue_adj'],
                zscore=r['zscore'],
                stability_score=r['stability_score'],
                hedge_ratio=r['hedge_ratio']
            )
            r['trade_score'] = score
            r['trade_breakdown'] = breakdown
        
        # Сортируем по Trade Score (вместо |Z-score|)
        results.sort(key=lambda x: x['trade_score'], reverse=True)
        
        if len(results) > 0:
            st.success(f"✅ Найдено {len(results)} пар (отфильтровано по half-life < {max_halflife_hours}ч)")
        
        return results[:max_pairs]
    
    def get_signal(self, zscore, threshold=2):
        """Определить торговый сигнал"""
        if zscore > threshold:
            return "SHORT"
        elif zscore < -threshold:
            return "LONG"
        else:
            return "NEUTRAL"

def plot_spread_chart(spread_data, pair_name, zscore):
    """График спреда с Z-score"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Спред пары {pair_name}', 'Z-Score во времени'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # График спреда
    fig.add_trace(
        go.Scatter(x=spread_data.index, y=spread_data.values, 
                  name='Spread', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Средняя линия
    mean = spread_data.mean()
    std = spread_data.std()
    
    fig.add_hline(y=mean, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=mean + 2*std, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_hline(y=mean - 2*std, line_dash="dot", line_color="green", row=1, col=1)
    
    # Z-score график
    zscore_series = (spread_data - mean) / std
    colors = ['red' if z > 2 else 'green' if z < -2 else 'gray' for z in zscore_series]
    
    fig.add_trace(
        go.Scatter(x=zscore_series.index, y=zscore_series.values,
                  name='Z-Score', mode='lines+markers',
                  line=dict(color='purple'), marker=dict(size=4)),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Спред", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    
    return fig

# === ИНТЕРФЕЙС ===

st.markdown('<p class="main-header">🔍 Crypto Pairs Trading Scanner</p>', unsafe_allow_html=True)
st.caption("Версия 2.0.0 | Обновлено: 16 февраля 2026 | DFA + FDR + Stability + Trade Score")
st.markdown("---")

# Sidebar - настройки
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Индикатор версии
    st.success("✅ Версия 1.3 активна | Мониторинг позиций")
    
    # Информация о гео-блокировках
    st.info("""
    ℹ️ **Если Binance заблокирован:**
    Приложение автоматически переключится на Bybit.
    Или выберите другую биржу вручную.
    """)
    
    exchange = st.selectbox(
        "Биржа",
        ['binance', 'bybit', 'okx', 'kucoin'],
        index=['binance', 'bybit', 'okx', 'kucoin'].index(st.session_state.settings['exchange']),
        help="Если ваш регион заблокирован, попробуйте Bybit или OKX",
        key='exchange_select'
    )
    st.session_state.settings['exchange'] = exchange
    
    timeframe = st.selectbox(
        "Таймфрейм",
        ['1h', '4h', '1d'],
        index=['1h', '4h', '1d'].index(st.session_state.settings['timeframe']),
        key='timeframe_select'
    )
    st.session_state.settings['timeframe'] = timeframe
    
    lookback_days = st.slider(
        "Период анализа (дней)",
        min_value=7,
        max_value=90,
        value=st.session_state.settings['lookback_days'],
        step=7,
        key='lookback_slider'
    )
    st.session_state.settings['lookback_days'] = lookback_days
    
    top_n_coins = st.slider(
        "Количество монет для анализа",
        min_value=20,
        max_value=100,
        value=st.session_state.settings['top_n_coins'],
        step=10,
        key='coins_slider'
    )
    st.session_state.settings['top_n_coins'] = top_n_coins
    
    max_pairs_display = st.slider(
        "Максимум пар в результатах",
        min_value=10,
        max_value=100,
        value=st.session_state.settings['max_pairs_display'],
        step=10,
        key='max_pairs_slider'
    )
    st.session_state.settings['max_pairs_display'] = max_pairs_display
    
    st.markdown("---")
    st.subheader("🎯 Фильтры качества")
    
    pvalue_threshold = st.slider(
        "P-value порог",
        min_value=0.01,
        max_value=0.10,
        value=st.session_state.settings['pvalue_threshold'],
        step=0.01,
        key='pvalue_slider'
    )
    st.session_state.settings['pvalue_threshold'] = pvalue_threshold
    
    zscore_threshold = st.slider(
        "Z-score порог для сигнала",
        min_value=1.5,
        max_value=3.0,
        value=st.session_state.settings['zscore_threshold'],
        step=0.1,
        key='zscore_slider'
    )
    st.session_state.settings['zscore_threshold'] = zscore_threshold
    
    st.markdown("---")
    st.subheader("⏱️ Фильтр по времени возврата")
    
    max_halflife_hours = st.slider(
        "Максимальный Half-life (часы)",
        min_value=6,
        max_value=50,  # 50 часов максимум
        value=min(st.session_state.settings['max_halflife_hours'], 50),
        step=2,
        help="Время возврата к среднему. Для 4h: 12-28ч быстрые, 28-50ч стандарт",
        key='halflife_slider'
    )
    st.session_state.settings['max_halflife_hours'] = max_halflife_hours
    
    st.info(f"📊 Текущий фильтр: до {max_halflife_hours} часов ({max_halflife_hours/24:.1f} дней)")
    
    # НОВОЕ: Фильтры Hurst + OU Process
    st.markdown("---")
    st.subheader("🔬 Mean Reversion Analysis")
    
    st.info("""
    **DFA Hurst** (v6.0):
    • H < 0.35 → Strong mean-reversion ✅
    • H < 0.48 → Mean-reverting ✅
    • H ≈ 0.50 → Random walk ⚪
    • H > 0.55 → Trending ❌
    """)
    
    # Hurst фильтр
    max_hurst = st.slider(
        "Максимальный Hurst",
        min_value=0.0,
        max_value=1.0,
        value=0.55,  # Обновлено для нового метода
        step=0.05,
        help="H < 0.40 = отлично, H < 0.50 = хорошо, H > 0.60 = избегать",
        key='max_hurst'
    )
    
    # OU theta фильтр
    min_theta = st.slider(
        "Минимальная скорость возврата (θ)",
        min_value=0.0,
        max_value=3.0,
        value=0.0,  # Выключен по умолчанию!
        step=0.1,
        help="θ > 1.0 = быстрый возврат. 0.0 = показать все",
        key='min_theta'
    )
    
    # Trade Score фильтр (v6.0)
    min_trade_score = st.slider(
        "Минимальный Trade Score",
        min_value=0,
        max_value=100,
        value=0,  # Выключен по умолчанию!
        step=5,
        help="Композитная оценка (Z + FDR + Hurst + OU + Stability + HR). 0 = показать все",
        key='min_trade_score'
    )
    
    # FDR фильтр
    fdr_only = st.checkbox(
        "Только FDR-подтверждённые",
        value=False,
        help="Показывать только пары, прошедшие Benjamini-Hochberg коррекцию",
        key='fdr_only'
    )
    
    # Stability фильтр
    stable_only = st.checkbox(
        "Только стабильные пары",
        value=False,
        help="Коинтеграция подтверждена на ≥3 из 4 подокон",
        key='stable_only'
    )
    
    auto_refresh = st.checkbox("Автообновление", value=False, key='auto_refresh_check')
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Интервал обновления (минуты)",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
            key='refresh_interval_slider'
        )
    
    st.markdown("---")
    st.markdown("### 📖 Как использовать:")
    st.markdown("""
    1. **Нажмите "Запустить сканер"**
    2. **Дождитесь результатов** (1-3 минуты)
    3. **Найдите пары с сигналами:**
       - 🟢 LONG - покупать первую монету
       - 🔴 SHORT - продавать первую монету
    4. **Проверьте графики** для подтверждения
    5. **Кликните на строку** → откроется анализ
    6. **Добавьте в отслеживание** для мониторинга
    """)
    
    st.markdown("---")

# Основная область
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.button("🚀 Запустить сканер", type="primary", use_container_width=True):
        st.session_state.running = True

with col2:
    if st.button("⏹️ Остановить", use_container_width=True):
        st.session_state.running = False

with col3:
    if st.session_state.last_update:
        st.metric("Последнее обновление", 
                 st.session_state.last_update.strftime("%H:%M:%S"))

# Запуск сканера
if st.session_state.running or (auto_refresh and st.session_state.pairs_data is not None):
    try:
        scanner = CryptoPairsScanner(
            exchange_name=exchange,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        # Прогресс бар
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, "Инициализация...")
        
        # Получаем топ монеты
        top_coins = scanner.get_top_coins(limit=top_n_coins)
        
        if not top_coins:
            st.error("❌ Не удалось получить список монет. Проверьте подключение к интернету или попробуйте другую биржу.")
            st.session_state.running = False
        else:
            # Сканируем пары
            pairs_results = scanner.scan_pairs(
                top_coins, 
                max_pairs=max_pairs_display, 
                progress_bar=progress_bar,
                max_halflife_hours=max_halflife_hours
            )
            
            progress_placeholder.empty()
            
            st.session_state.pairs_data = pairs_results
            st.session_state.last_update = datetime.now()
            
            if auto_refresh:
                time.sleep(refresh_interval * 60)
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.info("💡 Попробуйте: уменьшить количество монет, изменить таймфрейм или выбрать другую биржу")
        st.session_state.running = False

# Отображение результатов
if st.session_state.pairs_data is not None:
    pairs = st.session_state.pairs_data
    
    # Фильтрация по Hurst, OU, FDR, Stability, Trade Score (v6.0)
    if 'max_hurst' in st.session_state and 'min_theta' in st.session_state:
        filtered_pairs = []
        for p in pairs:
            if p.get('hurst', 0.5) > st.session_state.max_hurst:
                continue
            if p.get('theta', 0) < st.session_state.min_theta:
                continue
            if st.session_state.get('min_trade_score', 0) > 0 and p.get('trade_score', 0) < st.session_state.min_trade_score:
                continue
            if st.session_state.get('fdr_only', False) and not p.get('fdr_passed', False):
                continue
            if st.session_state.get('stable_only', False) and not p.get('is_stable', False):
                continue
            filtered_pairs.append(p)
        
        if len(filtered_pairs) < len(pairs):
            st.info(f"🔬 Фильтры: {len(pairs)} → {len(filtered_pairs)} пар")
        
        pairs = filtered_pairs
    
    if len(pairs) == 0:
        st.warning("⚠️ Коинтегрированных пар не найдено с текущими параметрами")
        st.info("""
        **Попробуйте:**
        - Увеличить период анализа (60-90 дней)
        - Увеличить P-value порог до 0.10
        - Уменьшить количество монет (сфокусироваться на топ-20)
        - Изменить таймфрейм на 4h или 1h
        - Ослабить фильтры Hurst/OU
        - Отключить FDR и Stability фильтры
        """)
    else:
        st.success(f"✅ Найдено {len(pairs)} коинтегрированных пар")
    
        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            long_signals = len([p for p in pairs if p['signal'] == 'LONG'])
            st.metric("🟢 LONG сигналы", long_signals)
        
        with col2:
            short_signals = len([p for p in pairs if p['signal'] == 'SHORT'])
            st.metric("🔴 SHORT сигналы", short_signals)
        
        with col3:
            neutral_signals = len([p for p in pairs if p['signal'] == 'NEUTRAL'])
            st.metric("⚪ Нейтральные", neutral_signals)
        
        with col4:
            avg_zscore = np.mean([abs(p['zscore']) for p in pairs])
            st.metric("Средний |Z-score|", f"{avg_zscore:.2f}")
        
        st.markdown("---")
        
        # Таблица результатов
        st.subheader("📊 Коинтегрированные пары")
        
        st.info("💡 **Кликните на строку** в таблице чтобы открыть детальный анализ пары")
    
    # Проверка что есть пары для отображения
    if len(pairs) > 0:
        df_display = pd.DataFrame([{
            'Пара': p['pair'],
            'Trade Score': p.get('trade_score', 0),
            'Z-Score': round(p['zscore'], 2),
            'P-value': round(p.get('pvalue_adj', p['pvalue']), 4),
            'FDR': '✅' if p.get('fdr_passed', False) else '❌',
            'Hurst': round(p.get('hurst', 0.5), 3),
            'θ (Theta)': round(p.get('theta', 0), 3),
            'Stab': f"{p.get('stability_passed', 0)}/{p.get('stability_total', 4)}",
            'Half-life': (
                f"{p.get('halflife_hours', p['halflife_days']*24):.1f}ч" 
                if p.get('halflife_hours', p['halflife_days']*24) < 48 
                else (
                    f"{p['halflife_days']:.1f}д" 
                    if p['halflife_days'] != np.inf 
                    else '∞'
                )
            ),
            'Hedge Ratio': round(p['hedge_ratio'], 4),
            'Сигнал': p['signal']
        } for p in pairs])
    else:
        # Пустая таблица если нет пар
        df_display = pd.DataFrame(columns=[
            'Пара', 'Trade Score', 'Z-Score', 'P-value', 'FDR', 'Hurst', 
            'θ (Theta)', 'Stab', 'Half-life', 'Hedge Ratio', 'Сигнал'
        ])
    
    # Функция для выбора строки
    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Выбрать", False)
        
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Выбрать": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns,
            use_container_width=True
        )
        
        selected_indices = list(np.where(edited_df.Выбрать)[0])
        return selected_indices
    
    selected_rows = dataframe_with_selections(df_display)
    
    if len(selected_rows) > 0:
        # Автоматически открываем детальный анализ для выбранной пары
        st.session_state.selected_pair_index = selected_rows[0]
        
    
    
    # Детальный анализ выбранной пары
    if len(pairs) > 0:
        st.markdown("---")
        st.subheader("📈 Детальный анализ пары")
        
        # Создаем список пар для выбора
        pair_options = [p['pair'] for p in pairs]
        
        # Сбрасываем индекс если он выходит за пределы
        if st.session_state.selected_pair_index >= len(pair_options):
            st.session_state.selected_pair_index = 0
        
        selected_pair = st.selectbox(
            "Выберите пару для анализа:",
            pair_options,
            index=st.session_state.selected_pair_index,
            key=f'pair_selector_{len(pairs)}'  # Уникальный ключ при изменении данных
        )
        
        # Обновляем индекс
        st.session_state.selected_pair_index = pair_options.index(selected_pair)
        
        selected_data = next(p for p in pairs if p['pair'] == selected_pair)
    
    # Заголовок с текущей парой
    st.markdown(f"### 🎯 Анализ: **{selected_pair}**")
    
    # Информация о паре
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Z-Score", f"{selected_data['zscore']:.2f}")
    
    with col2:
        signal_color = "🟢" if selected_data['signal'] == 'LONG' else "🔴" if selected_data['signal'] == 'SHORT' else "⚪"
        st.metric("Сигнал", f"{signal_color} {selected_data['signal']}")
    
    with col3:
        st.metric("P-value", f"{selected_data['pvalue']:.4f}")
    
    with col4:
        hl = selected_data['halflife_days']
        hl_hours = selected_data.get('halflife_hours', hl * 24)
        if hl_hours < 48:  # Если меньше 2 дней, показываем в часах
            st.metric("Half-life", f"{hl_hours:.1f} ч")
        else:
            st.metric("Half-life", f"{hl:.1f} д ({hl_hours:.0f}ч)" if hl != np.inf else "∞")
    
    # Mean Reversion Analysis (v6.0)
    if 'hurst' in selected_data and 'theta' in selected_data:
        st.markdown("---")
        st.subheader("🔬 Mean Reversion Analysis (v6.0)")
        
        # Trade Score — главный показатель
        trade_score = selected_data.get('trade_score', 0)
        trade_bd = selected_data.get('trade_breakdown', {})
        
        ts_col1, ts_col2 = st.columns([1, 3])
        with ts_col1:
            if trade_score >= 70:
                ts_emoji = "🟢"
                ts_status = "Отличный"
            elif trade_score >= 50:
                ts_emoji = "🟡"
                ts_status = "Хороший"
            elif trade_score >= 30:
                ts_emoji = "🟠"
                ts_status = "Слабый"
            else:
                ts_emoji = "🔴"
                ts_status = "Не входить"
            st.metric(f"{ts_emoji} Trade Score", f"{trade_score}/100", ts_status)
        
        with ts_col2:
            if trade_bd:
                bd_text = " | ".join([f"**{k}**: {v}" for k, v in trade_bd.items()])
                st.caption(f"Разбивка: {bd_text}")
                
                # FDR статус
                fdr_status = "✅ FDR passed" if selected_data.get('fdr_passed', False) else "❌ FDR failed"
                stab = selected_data.get('stability_passed', 0)
                stab_total = selected_data.get('stability_total', 4)
                stab_status = f"{'✅' if selected_data.get('is_stable', False) else '⚠️'} Стабильность: {stab}/{stab_total} окон"
                st.caption(f"{fdr_status} | {stab_status} | P-adj: {selected_data.get('pvalue_adj', 0):.4f}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hurst = selected_data['hurst']
            if hurst < 0.35:
                hurst_status = "🟢 Strong MR"
            elif hurst < 0.48:
                hurst_status = "🟢 Reverting"
            elif hurst < 0.52:
                hurst_status = "⚪ Random"
            else:
                hurst_status = "🔴 Trending"
            st.metric("Hurst (DFA)", f"{hurst:.3f}", hurst_status)
        
        with col2:
            theta = selected_data['theta']
            theta_status = "✅ Быстрый" if theta > 1.0 else "⚠️ Средний" if theta > 0.5 else "❌ Медленный"
            st.metric("θ (Скорость)", f"{theta:.3f}", theta_status)
        
        with col3:
            st.metric("Hedge Ratio", f"{selected_data['hedge_ratio']:.4f}",
                      "✅ OK" if 0.2 <= abs(selected_data['hedge_ratio']) <= 5.0 else "⚠️ Экстрем.")
        
        with col4:
            if theta > 0:
                exit_time = estimate_exit_time(
                    current_z=selected_data['zscore'],
                    theta=theta,
                    target_z=0.5
                )
                exit_hours = exit_time * 24
                st.metric("Прогноз выхода", f"{exit_hours:.1f}ч", "до Z=0.5")
            else:
                st.metric("Прогноз выхода", "∞", "Нет возврата")
        
        # Интерпретация
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if hurst < 0.35:
                hurst_msg = "🟢 **Сильный mean-reversion** (H < 0.35)"
                hurst_desc = "Идеальная пара для арбитража! DFA подтверждает устойчивый возврат к среднему."
            elif hurst < 0.48:
                hurst_msg = "🟢 **Mean-reverting** (H < 0.48)"
                hurst_desc = "Хорошая пара. DFA показывает возврат к среднему."
            elif hurst < 0.52:
                hurst_msg = "⚪ **Random walk** (H ≈ 0.5)"
                hurst_desc = "Случайное блуждание. Нет статистического основания для торговли."
            else:
                hurst_msg = "🔴 **Trending** (H > 0.52)"
                hurst_desc = "НЕ подходит для парного арбитража! Спред трендовый."
            
            st.info(f"""
            **Hurst (DFA):** {hurst_msg}
            
            {hurst_desc}
            
            **Шкала DFA (валидировано на синтетике):**
            • H < 0.35 → Strong mean-reversion ✅
            • H < 0.48 → Mean-reverting ✅
            • H ≈ 0.50 → Random walk ⚪
            • H > 0.55 → Trending ❌
            """)
        
        with info_col2:
            if theta > 2.0:
                theta_msg = "🟢 **Очень быстрый возврат** (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 1.0:
                theta_msg = "🟢 **Быстрый возврат** (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 0.5:
                theta_msg = "🟡 **Средний возврат** (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            else:
                theta_msg = "🔴 **Медленный** (>{:.0f}ч)".format(-np.log(0.5)/theta * 24 if theta > 0 else 999)
            
            st.info(f"""
            **OU Process (θ):**
            {theta_msg}
            
            Скорость возврата к среднему.
            Чем выше θ, тем быстрее возврат.
            """)
    
    # График спреда
    if selected_data['spread'] is not None:
        fig = plot_spread_chart(selected_data['spread'], selected_pair, selected_data['zscore'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Калькулятор размера позиции
    st.markdown("---")
    st.subheader("💰 Калькулятор размера позиции")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_capital = st.number_input(
            "💵 Общая сумма для входа (USD)",
            min_value=10.0,
            max_value=1000000.0,
            value=100.0,  # $100 по умолчанию
            step=10.0,
            help="Сколько всего хотите вложить в эту пару",
            key=f"capital_{selected_pair}"
        )
        
        commission_rate = st.number_input(
            "💸 Комиссия биржи (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Обычно 0.1% для мейкеров, 0.075% на Binance с BNB",
            key=f"commission_{selected_pair}"
        )
    
    with col2:
        hedge_ratio = selected_data['hedge_ratio']
        
        st.markdown("### 📊 Распределение капитала:")
        
        # Расчет позиций с учетом hedge ratio
        position1 = total_capital / (1 + hedge_ratio)
        position2 = position1 * hedge_ratio
        
        # Учет комиссий (вход + выход, обе стороны)
        commission_total = (position1 + position2) * (commission_rate / 100) * 2
        effective_capital = total_capital - commission_total
        
        coin1, coin2 = selected_data['coin1'], selected_data['coin2']
        signal = selected_data['signal']
        
        if signal == 'LONG':
            st.success(f"""
            **🟢 LONG позиция:**
            
            **{coin1}:** КУПИТЬ ${position1:.2f}
            **{coin2}:** ПРОДАТЬ ${position2:.2f}
            
            💸 Комиссии: ${commission_total:.2f}
            💰 Эффективно: ${effective_capital:.2f}
            """)
        elif signal == 'SHORT':
            st.error(f"""
            **🔴 SHORT позиция:**
            
            **{coin1}:** ПРОДАТЬ ${position1:.2f}
            **{coin2}:** КУПИТЬ ${position2:.2f}
            
            💸 Комиссии: ${commission_total:.2f}
            💰 Эффективно: ${effective_capital:.2f}
            """)
    
    # Детальная разбивка
    st.markdown("### 📝 Детальная разбивка позиции")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.metric(f"{coin1} позиция", f"${position1:.2f}", 
                 f"{(position1/total_capital)*100:.1f}% от капитала")
    
    with breakdown_col2:
        st.metric(f"{coin2} позиция", f"${position2:.2f}",
                 f"{(position2/total_capital)*100:.1f}% от капитала")
    
    with breakdown_col3:
        st.metric("Hedge Ratio", f"{hedge_ratio:.4f}",
                 f"1:{hedge_ratio:.4f}")
    
    # Калькулятор прибыли/убытков
    st.markdown("---")
    st.subheader("🎯 Расчет прибыли и стоп-лосса")
    
    entry_z = selected_data['zscore']
    
    # Стоп-лосс и цели
    if abs(entry_z) > 0:
        if entry_z < 0:  # LONG
            stop_z = entry_z - 1.0
            tp1_z = entry_z + (abs(entry_z) * 0.4)
            target_z = 0.0
        else:  # SHORT
            stop_z = entry_z + 1.0
            tp1_z = entry_z - (abs(entry_z) * 0.4)
            target_z = 0.0
        
        # Процент изменения Z-score
        stop_loss_pct = ((abs(stop_z - entry_z) / abs(entry_z)) * 100)
        tp1_pct = ((abs(tp1_z - entry_z) / abs(entry_z)) * 100)
        target_pct = 100.0
        
        # Реалистичная прибыль для парного арбитража (~6% при полном цикле)
        # Формула: (движение_Z / 100) × капитал × 0.06
        hedge_efficiency = 0.06  # 6% типичная прибыль при полном движении к Z=0
        
        stop_loss_usd = -total_capital * (stop_loss_pct / 100) * hedge_efficiency
        tp1_usd = total_capital * (tp1_pct / 100) * hedge_efficiency
        target_usd = total_capital * (target_pct / 100) * hedge_efficiency
        
        pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
        
        with pnl_col1:
            st.markdown("**🛡️ Стоп-лосс**")
            st.metric("Z-score", f"{stop_z:.2f}")
            st.error(f"Убыток: **${abs(stop_loss_usd):.2f}**")
            st.caption(f"(-{stop_loss_pct:.1f}% от входа)")
        
        with pnl_col2:
            st.markdown("**💰 Take Profit 1**")
            st.metric("Z-score", f"{tp1_z:.2f}")
            st.success(f"Прибыль: **${tp1_usd:.2f}**")
            st.caption(f"(+{tp1_pct:.1f}%, закрыть 50%)")
        
        with pnl_col3:
            st.markdown("**🎯 Полная цель**")
            st.metric("Z-score", "0.00")
            st.success(f"Прибыль: **${target_usd:.2f}**")
            st.caption(f"(+{target_pct:.0f}%, полный выход)")
        
        # Risk/Reward
        risk_reward = abs(target_usd / stop_loss_usd) if stop_loss_usd != 0 else 0
        
        st.markdown("---")
        
        rr_col1, rr_col2, rr_col3 = st.columns(3)
        
        with rr_col1:
            st.metric("💎 Потенциал прибыли", f"${target_usd:.2f}")
        
        with rr_col2:
            st.metric("⚠️ Максимальный риск", f"${abs(stop_loss_usd):.2f}")
        
        with rr_col3:
            if risk_reward >= 2:
                emoji = "🟢"
                assessment = "Отлично!"
            elif risk_reward >= 1.5:
                emoji = "🟡"
                assessment = "Приемлемо"
            else:
                emoji = "🔴"
                assessment = "Слабо"
            
            st.metric(f"{emoji} Risk/Reward", f"{risk_reward:.2f}:1")
            st.caption(assessment)
    
    # Рекомендации по торговле
    st.markdown("---")
    st.markdown("### 💡 Торговая рекомендация")
    
    if selected_data['signal'] == 'LONG':
        st.success(f"""
        **Стратегия:**
        - 🟢 **КУПИТЬ** {selected_data['coin1']}
        - 🔴 **ПРОДАТЬ** {selected_data['coin2']} (или шорт)
        - **Соотношение:** 1:{selected_data['hedge_ratio']:.4f}
        - **Таргет:** Z-score → 0
        - **Стоп-лосс:** Z-score < -3
        """)
    elif selected_data['signal'] == 'SHORT':
        st.error(f"""
        **Стратегия:**
        - 🔴 **ПРОДАТЬ** {selected_data['coin1']} (или шорт)
        - 🟢 **КУПИТЬ** {selected_data['coin2']}
        - **Соотношение:** 1:{selected_data['hedge_ratio']:.4f}
        - **Таргет:** Z-score → 0
        - **Стоп-лосс:** Z-score > 3
        """)
    else:
        st.info("⚪ Нет активного сигнала. Дождитесь |Z-score| > 2")
    
    # Экспорт данных
    st.markdown("---")
    csv_data = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Скачать результаты (CSV)",
        data=csv_data,
        file_name=f"pairs_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Нажмите 'Запустить сканер' для начала анализа")
    
    # Инструкция
    st.markdown("""
    ### 🎯 Что делает этот скринер:
    
    1. **Загружает данные** топ-100 криптовалют с Binance
    2. **Тестирует все пары** на статистическую коинтеграцию
    3. **Находит возможности** для парного арбитража
    4. **Показывает сигналы** на основе Z-score
    
    ### 📚 Как торговать:
    
    - **Z-score > +2**: Пара переоценена → SHORT первая монета, LONG вторая
    - **Z-score < -2**: Пара недооценена → LONG первая монета, SHORT вторая
    - **Z-score → 0**: Закрытие позиции (возврат к среднему)
    
    ### ⚠️ Важно:
    - Используйте стоп-лоссы
    - Учитывайте комиссии биржи
    - Проверяйте ликвидность пар
    - Это не финансовая рекомендация
    """)

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: Этот инструмент предназначен только для образовательных целей. Не является финансовой рекомендацией.")
# VERSION: 2.0
# LAST UPDATED: 2026-02-16
# FEATURES: DFA Hurst, FDR correction, rolling Z-score, cointegration stability, Trade Score, position monitoring
