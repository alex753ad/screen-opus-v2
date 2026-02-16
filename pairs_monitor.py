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

# Импорт модуля mean reversion analysis v8.1
from mean_reversion_analysis import (
    calculate_hurst_exponent,
    calculate_rolling_zscore,
    calculate_ou_parameters,
    calculate_ou_score,
    calculate_quality_score,
    calculate_signal_score,
    calculate_trade_score,
    calculate_confidence,
    get_adaptive_signal,
    sanitize_pair,
    apply_fdr_correction,
    check_cointegration_stability,
    adf_test_spread,
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
        """Сканировать все пары (v7.0: 2-pass FDR + ADF + Confidence)"""
        
        # Загружаем данные
        st.info(f"📥 Загружаю данные для {len(coins)} монет...")
        price_data = {}
        for coin in coins:
            symbol = f"{coin}/USDT"
            prices = self.fetch_ohlcv(symbol)
            if prices is not None and len(prices) > 20:
                price_data[coin] = prices
            time.sleep(0.1)
        
        if len(price_data) < 2:
            st.error("❌ Недостаточно данных для анализа")
            return []
        
        total_combinations = len(price_data) * (len(price_data) - 1) // 2
        st.info(f"🔍 Фаза 1: Коинтеграция для {total_combinations} пар из {len(price_data)} монет...")
        processed = 0
        
        # ═══════ ФАЗА 1: Быстрый тест коинтеграции для ВСЕХ пар ═══════
        # Собираем ВСЕ p-values (ключевое исправление FDR!)
        all_pvalues = []
        candidates = []  # (coin1, coin2, result) для пар с p < 0.10
        
        for i, coin1 in enumerate(price_data.keys()):
            for coin2 in list(price_data.keys())[i+1:]:
                processed += 1
                if progress_bar:
                    progress_bar.progress(
                        processed / total_combinations * 0.5,  # Фаза 1 = 50%
                        f"Фаза 1: {processed}/{total_combinations}"
                    )
                
                result = self.test_cointegration(price_data[coin1], price_data[coin2])
                
                if result:
                    all_pvalues.append(result['pvalue'])
                    
                    # Сохраняем кандидатов (p < 0.10 для запаса)
                    halflife_hours = result['halflife'] * 24
                    if result['pvalue'] < 0.10 and halflife_hours <= max_halflife_hours:
                        candidates.append((coin1, coin2, result, len(all_pvalues) - 1))
                else:
                    all_pvalues.append(1.0)  # Не удалось — p=1
        
        # ═══════ FDR на ВСЕХ p-values ═══════
        if len(all_pvalues) == 0:
            return []
        
        adj_pvalues, fdr_rejected = apply_fdr_correction(all_pvalues, alpha=0.05)
        
        total_fdr_passed = int(np.sum(fdr_rejected))
        st.info(f"🔬 FDR: {total_fdr_passed} из {len(all_pvalues)} пар прошли (α=0.05)")
        
        # ═══════ ФАЗА 2: Дорогие метрики только для кандидатов ═══════
        st.info(f"🔍 Фаза 2: Детальный анализ {len(candidates)} кандидатов...")
        results = []
        dt = {'1h': 1/24, '4h': 1/6, '1d': 1}.get(self.timeframe, 1/6)
        
        for idx_c, (coin1, coin2, result, pval_idx) in enumerate(candidates):
            if progress_bar:
                progress_bar.progress(
                    0.5 + (idx_c + 1) / len(candidates) * 0.5,
                    f"Фаза 2: {idx_c + 1}/{len(candidates)}"
                )
            
            fdr_passed = bool(fdr_rejected[pval_idx])
            pvalue_adj = float(adj_pvalues[pval_idx])
            
            # Hurst (DFA)
            hurst = calculate_hurst_exponent(result['spread'])
            hurst_is_fallback = (hurst == 0.5)
            
            # OU
            ou_params = calculate_ou_parameters(result['spread'], dt=dt)
            ou_score = calculate_ou_score(ou_params, hurst)
            is_valid, reason = validate_ou_quality(ou_params, hurst)
            
            # Stability
            stability = check_cointegration_stability(
                price_data[coin1].values, price_data[coin2].values
            )
            
            # [v8.1] Sanitizer — жёсткие исключения
            san_ok, san_reason = sanitize_pair(
                hedge_ratio=result['hedge_ratio'],
                stability_passed=stability['windows_passed'],
                stability_total=stability['total_windows'],
                zscore=result['zscore']
            )
            if not san_ok:
                continue
            
            # [NEW] ADF-тест спреда
            adf = adf_test_spread(result['spread'])
            
            # [NEW] Confidence
            confidence, conf_checks, conf_total = calculate_confidence(
                hurst=hurst,
                stability_score=stability['stability_score'],
                fdr_passed=fdr_passed,
                adf_passed=adf['is_stationary'],
                zscore=result['zscore'],
                hedge_ratio=result['hedge_ratio'],
                hurst_is_fallback=hurst_is_fallback
            )
            
            # [D-B] Quality Score (пара)
            q_score, q_breakdown = calculate_quality_score(
                hurst=hurst,
                ou_params=ou_params,
                pvalue_adj=pvalue_adj,
                stability_score=stability['stability_score'],
                hedge_ratio=result['hedge_ratio'],
                adf_passed=adf['is_stationary'],
                hurst_is_fallback=hurst_is_fallback
            )
            
            # [v8.1] Signal Score (capped by Quality)
            s_score, s_breakdown = calculate_signal_score(
                zscore=result['zscore'],
                ou_params=ou_params,
                confidence=confidence,
                quality_score=q_score
            )
            
            # [v8.1] Adaptive Signal (TF-aware)
            state, direction, threshold = get_adaptive_signal(
                zscore=result['zscore'],
                confidence=confidence,
                quality_score=q_score,
                timeframe=self.timeframe
            )
            
            halflife_hours = result['halflife'] * 24
            
            results.append({
                'pair': f"{coin1}/{coin2}",
                'coin1': coin1,
                'coin2': coin2,
                'pvalue': result['pvalue'],
                'pvalue_adj': pvalue_adj,
                'fdr_passed': fdr_passed,
                'zscore': result['zscore'],
                'zscore_series': result.get('zscore_series'),
                'hedge_ratio': result['hedge_ratio'],
                'intercept': result.get('intercept', 0.0),
                'halflife_days': result['halflife'],
                'halflife_hours': halflife_hours,
                'spread': result['spread'],
                # v8: adaptive signal вместо get_signal
                'signal': state,          # SIGNAL/READY/WATCH/NEUTRAL
                'direction': direction,    # LONG/SHORT/NONE
                'threshold': threshold,    # порог для этой пары
                'hurst': hurst,
                'hurst_is_fallback': hurst_is_fallback,
                'theta': ou_params['theta'] if ou_params else 0,
                'mu': ou_params['mu'] if ou_params else 0,
                'sigma': ou_params['sigma'] if ou_params else 0,
                'halflife_ou': ou_params['halflife_ou'] * 24 if ou_params else 999,
                'ou_score': ou_score,
                'ou_valid': is_valid,
                'ou_reason': reason,
                'stability_score': stability['stability_score'],
                'stability_passed': stability['windows_passed'],
                'stability_total': stability['total_windows'],
                'is_stable': stability['is_stable'],
                'adf_pvalue': adf['adf_pvalue'],
                'adf_passed': adf['is_stationary'],
                'quality_score': q_score,
                'quality_breakdown': q_breakdown,
                'signal_score': s_score,
                'signal_breakdown': s_breakdown,
                'trade_score': q_score,       # legacy compat
                'trade_breakdown': q_breakdown,
                'confidence': confidence,
                'conf_checks': conf_checks,
                'conf_total': conf_total,
            })
        
        # Сортируем: сначала по Signal (SIGNAL > READY > WATCH > NEUTRAL), потом по Quality
        signal_order = {'SIGNAL': 0, 'READY': 1, 'WATCH': 2, 'NEUTRAL': 3}
        results.sort(key=lambda x: (signal_order.get(x['signal'], 4), -x['quality_score']))
        
        if len(results) > 0:
            st.success(f"✅ Найдено {len(results)} пар (FDR: {total_fdr_passed} подтверждены)")
        
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
st.caption("Версия 3.1.0 | 16 февраля 2026 | Sanitizers + TF-thresholds + Quality/Signal + DFA + ADF + FDR")
st.markdown("---")

# Sidebar - настройки
with st.sidebar:
    st.header("⚙️ Настройки")
    
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
    
    # Quality Score фильтр (v8.0)
    min_quality = st.slider(
        "Мин. Quality Score",
        min_value=0, max_value=100, value=0, step=5,
        help="Качество пары (FDR + Stability + Hurst + ADF + HR). 0 = все",
        key='min_quality'
    )
    
    # Signal state фильтр
    signal_filter = st.multiselect(
        "Показывать статусы",
        options=["SIGNAL", "READY", "WATCH", "NEUTRAL"],
        default=["SIGNAL", "READY", "WATCH", "NEUTRAL"],
        help="SIGNAL=вход, READY=почти, WATCH=мониторинг",
        key='signal_filter'
    )
    
    # FDR фильтр
    fdr_only = st.checkbox(
        "Только FDR-подтверждённые",
        value=False,
        help="Только пары, прошедшие Benjamini-Hochberg",
        key='fdr_only'
    )
    
    # Stability фильтр
    stable_only = st.checkbox(
        "Только стабильные пары",
        value=False,
        help="Коинтеграция ≥3/4 подокон",
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
    
    # Фильтрация v8.0
    if 'max_hurst' in st.session_state and 'min_theta' in st.session_state:
        filtered_pairs = []
        for p in pairs:
            if p.get('hurst', 0.5) > st.session_state.max_hurst:
                continue
            if p.get('theta', 0) < st.session_state.min_theta:
                continue
            if st.session_state.get('min_quality', 0) > 0 and p.get('quality_score', 0) < st.session_state.min_quality:
                continue
            if st.session_state.get('signal_filter') and p.get('signal', 'NEUTRAL') not in st.session_state.signal_filter:
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
    
        # Метрики v8
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            signals = len([p for p in pairs if p['signal'] == 'SIGNAL'])
            st.metric("🔴 SIGNAL", signals)
        
        with col2:
            ready = len([p for p in pairs if p['signal'] == 'READY'])
            st.metric("🟡 READY", ready)
        
        with col3:
            watch = len([p for p in pairs if p['signal'] == 'WATCH'])
            st.metric("👁 WATCH", watch)
        
        with col4:
            neutral = len([p for p in pairs if p['signal'] == 'NEUTRAL'])
            st.metric("⚪ NEUTRAL", neutral)
        
        with col5:
            high_q = len([p for p in pairs if p.get('confidence') == 'HIGH'])
            st.metric("⭐ HIGH conf", high_q)
        
        st.markdown("---")
        
        # Таблица результатов
        st.subheader("📊 Коинтегрированные пары")
        
        st.info("💡 **Кликните на строку** в таблице чтобы открыть детальный анализ пары")
    
    # Проверка что есть пары для отображения
    if len(pairs) > 0:
        df_display = pd.DataFrame([{
            'Пара': p['pair'],
            'Статус': p['signal'],
            'Dir': p.get('direction', ''),
            'Q': p.get('quality_score', 0),
            'S': p.get('signal_score', 0),
            'Conf': p.get('confidence', '?'),
            'Z': round(p['zscore'], 2),
            'Thr': p.get('threshold', 2.0),
            'FDR': '✅' if p.get('fdr_passed', False) else '❌',
            'ADF': '✅' if p.get('adf_passed', False) else '❌',
            'Hurst': round(p.get('hurst', 0.5), 3),
            'Stab': f"{p.get('stability_passed', 0)}/{p.get('stability_total', 4)}",
            'HL': (
                f"{p.get('halflife_hours', p['halflife_days']*24):.1f}ч" 
                if p.get('halflife_hours', p['halflife_days']*24) < 48 
                else '∞'
            ),
            'HR': round(p['hedge_ratio'], 4),
        } for p in pairs])
    else:
        # Пустая таблица если нет пар
        df_display = pd.DataFrame(columns=[
            'Пара', 'Статус', 'Dir', 'Q', 'S', 'Conf', 'Z', 'Thr',
            'FDR', 'ADF', 'Hurst', 'Stab', 'HL', 'HR'
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
        st.session_state.selected_pair_index = selected_rows[0]
    
    # Детальный анализ выбранной пары
    if len(pairs) > 0:
        st.markdown("---")
        st.subheader("📈 Детальный анализ пары")
        
        pair_options = [p['pair'] for p in pairs]
        
        # Ограничиваем индекс
        if st.session_state.selected_pair_index >= len(pair_options):
            st.session_state.selected_pair_index = 0
        
        # Selectbox с index из session_state (обновляется по checkbox)
        selected_pair = st.selectbox(
            "Выберите пару для анализа:",
            pair_options,
            index=st.session_state.selected_pair_index,
            key='pair_selector_main'
        )
        
        # Синхронизируем обратно
        st.session_state.selected_pair_index = pair_options.index(selected_pair)
        
        selected_data = next(p for p in pairs if p['pair'] == selected_pair)
    
    # ═══════ ЗАГОЛОВОК С АДАПТИВНЫМ СИГНАЛОМ ═══════
    state = selected_data.get('signal', 'NEUTRAL')
    direction = selected_data.get('direction', 'NONE')
    conf = selected_data.get('confidence', '?')
    threshold = selected_data.get('threshold', 2.0)
    
    state_emoji = {'SIGNAL': '🔴', 'READY': '🟡', 'WATCH': '👁', 'NEUTRAL': '⚪'}.get(state, '⚪')
    conf_emoji = {'HIGH': '⭐', 'MEDIUM': '🔵', 'LOW': '⚫'}.get(conf, '⚫')
    dir_emoji = {'LONG': '🟢↑', 'SHORT': '🔴↓', 'NONE': ''}.get(direction, '')
    
    st.markdown(f"### {state_emoji} **{state}** {dir_emoji} | {conf_emoji} {conf} | **{selected_pair}**")
    
    if state == 'SIGNAL':
        st.success(f"🎯 **ВХОД {direction}** | Z={selected_data['zscore']:.2f} | Порог для этой пары: |Z| ≥ {threshold}")
    elif state == 'READY':
        st.info(f"⏳ **ГОТОВНОСТЬ {direction}** | Z={selected_data['zscore']:.2f} | До порога ({threshold}): {abs(threshold - abs(selected_data['zscore'])):.2f}")
    elif state == 'WATCH':
        st.info(f"👁 **МОНИТОРИНГ** | Z={selected_data['zscore']:.2f} | Порог: {threshold}")
    
    # ⚠️ Предупреждения
    warnings_list = []
    if selected_data.get('hurst_is_fallback', False):
        warnings_list.append("⚠️ Hurst = 0.5 (DFA fallback — данных недостаточно)")
    if abs(selected_data['zscore']) > 5:
        warnings_list.append(f"⚠️ |Z| = {abs(selected_data['zscore']):.1f} > 5 — аномалия")
    if not selected_data.get('fdr_passed', False):
        warnings_list.append("⚠️ FDR не пройден — коинтеграция ненадёжна")
    if not selected_data.get('adf_passed', False):
        warnings_list.append("⚠️ ADF: спред нестационарен")
    if warnings_list:
        st.warning("\n".join(warnings_list))
    
    # ═══════ DUAL SCORE: Quality + Signal ═══════
    q_score = selected_data.get('quality_score', 0)
    s_score = selected_data.get('signal_score', 0)
    q_bd = selected_data.get('quality_breakdown', {})
    s_bd = selected_data.get('signal_breakdown', {})
    
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        q_emoji = "🟢" if q_score >= 60 else "🟡" if q_score >= 40 else "🔴"
        st.metric(f"{q_emoji} Quality Score", f"{q_score}/100", 
                  "Надёжность пары")
        if q_bd:
            st.caption(" | ".join([f"{k}:{v}" for k, v in q_bd.items()]))
    
    with score_col2:
        s_emoji = "🟢" if s_score >= 60 else "🟡" if s_score >= 30 else "⚪"
        st.metric(f"{s_emoji} Signal Score", f"{s_score}/100",
                  "Момент входа")
        if s_bd:
            st.caption(" | ".join([f"{k}:{v}" for k, v in s_bd.items()]))
    
    # ═══════ МЕТРИКИ ═══════
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Z-Score", f"{selected_data['zscore']:.2f}")
    with col2:
        st.metric("P-adj", f"{selected_data.get('pvalue_adj', selected_data['pvalue']):.4f}")
    with col3:
        hl_hours = selected_data.get('halflife_hours', selected_data['halflife_days'] * 24)
        st.metric("Half-life", f"{hl_hours:.1f}ч" if hl_hours < 48 else "∞")
    with col4:
        st.metric("Confidence", f"{conf} ({selected_data.get('conf_checks', 0)}/{selected_data.get('conf_total', 6)})")
    with col5:
        st.metric("Порог Z", f"±{threshold}")
    
    # ═══════ MEAN REVERSION ANALYSIS v8.0 ═══════
    if 'hurst' in selected_data and 'theta' in selected_data:
        st.markdown("---")
        st.subheader("🔬 Mean Reversion Analysis (v8.0)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hurst = selected_data['hurst']
            if selected_data.get('hurst_is_fallback', False):
                h_st = "⚠️ Fallback"
            elif hurst < 0.35:
                h_st = "🟢 Strong MR"
            elif hurst < 0.48:
                h_st = "🟢 Reverting"
            elif hurst < 0.52:
                h_st = "⚪ Random"
            else:
                h_st = "🔴 Trending"
            st.metric("Hurst (DFA)", f"{hurst:.3f}", h_st)
        
        with col2:
            theta = selected_data['theta']
            t_st = "✅ Быстрый" if theta > 1.0 else "⚠️ Средний" if theta > 0.5 else "❌ Медленный"
            st.metric("θ (Скорость)", f"{theta:.3f}", t_st)
        
        with col3:
            hr = selected_data['hedge_ratio']
            hr_st = "✅ OK" if 0.2 <= abs(hr) <= 5.0 else "⚠️ Экстрем."
            st.metric("Hedge Ratio", f"{hr:.4f}", hr_st)
        
        with col4:
            if theta > 0:
                exit_time = estimate_exit_time(
                    current_z=selected_data['zscore'], theta=theta, target_z=0.5
                )
                st.metric("Прогноз", f"{exit_time * 24:.1f}ч", "до Z=0.5")
            else:
                st.metric("Прогноз", "∞", "Нет возврата")
        
        # Проверки
        checks_col1, checks_col2 = st.columns(2)
        with checks_col1:
            fdr_s = "✅" if selected_data.get('fdr_passed', False) else "❌"
            adf_s = "✅" if selected_data.get('adf_passed', False) else "❌"
            stab = f"{selected_data.get('stability_passed', 0)}/{selected_data.get('stability_total', 4)}"
            stab_e = "✅" if selected_data.get('is_stable', False) else "⚠️"
            st.info(f"""
            **Проверки:**
            {fdr_s} FDR (p-adj={selected_data.get('pvalue_adj', 0):.4f})
            {adf_s} ADF (p={selected_data.get('adf_pvalue', 1.0):.4f})
            {stab_e} Стабильность: {stab} окон
            """)
        
        with checks_col2:
            if theta > 2.0:
                t_msg = "🟢 Очень быстрый (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 1.0:
                t_msg = "🟢 Быстрый (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            elif theta > 0.5:
                t_msg = "🟡 Средний (~{:.1f}ч)".format(-np.log(0.5)/theta * 24)
            else:
                t_msg = "🔴 Медленный"
            st.info(f"""
            **OU Process:** {t_msg}
            
            **Adaptive порог:** |Z| ≥ {threshold}
            ({conf} confidence → {'сниженный' if threshold < 2.0 else 'стандартный'} порог)
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
# VERSION: 3.0
# LAST UPDATED: 2026-02-16
# FEATURES: Quality/Signal dual score, adaptive Z-thresholds, SIGNAL/READY/WATCH states, HR cutoff, DFA, ADF, FDR on all pairs, stability, confidence
