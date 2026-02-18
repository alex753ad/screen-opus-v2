"""
Модуль расчета Hurst Exponent и Ornstein-Uhlenbeck параметров
ВЕРСИЯ v10.5.0: Parallel download, HR cap↓30, DFA↑8, Conditional Z-cap, FDR gate

Дата: 18 февраля 2026

ИЗМЕНЕНИЯ v10.4.0:
  [FIX] sanitize_pair() — HR cap: 50 → 30 (APT/ZORA HR=42 выходил через фильтр)
  [FIX] DFA min_window: 4 → 8 (устраняет нестабильный Hurst на коротких рядах)
  [NEW] get_adaptive_signal() — fdr_passed параметр: FDR fail + Q<60 → max READY
  [NEW] get_adaptive_signal() — Z>4.5 условный: Q≥70 AND Stab≥3/4 → SIGNAL (EXTREME)
  [FIX] Quality score: HR>10 penalty усилен (0 вместо 2)
  
  v10.3: Stability gate (Stab<2/4 → WATCH), Z cutoff 4.5
  v10.2: Q gate↑40, HR ceiling, N_min↑50
  v10.1: Min-Q gate, HR uncertainty, N-bars hard gate
  v10.0: Adaptive Robust Z, Crossing Density, Correlation, Kalman HR
"""

import numpy as np
from scipy import stats


# =============================================================================
# HURST — DFA
# =============================================================================

def calculate_hurst_exponent(time_series, min_window=8):
    """DFA на инкрементах. Возвращает 0.5 при fallback.
    
    v10.4: min_window=8 (was 4). Меньшее значение давало нестабильные
    результаты на 100-300 свечах, что приводило к Hurst=0.085 на 35d
    и Hurst=0.5 (fallback) на 63d для одной и той же пары.
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    if n < 30:
        return 0.5

    increments = np.diff(ts)
    n_inc = len(increments)
    profile = np.cumsum(increments - np.mean(increments))

    max_window = n_inc // 4
    if max_window <= min_window:
        return 0.5

    num_points = min(20, max_window - min_window)
    if num_points < 4:
        return 0.5

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_points).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]
    if len(window_sizes) < 4:
        return 0.5

    fluctuations = []
    for w in window_sizes:
        n_seg = n_inc // w
        if n_seg < 2:
            continue
        f2_sum, count = 0.0, 0
        for seg in range(n_seg):
            segment = profile[seg * w:(seg + 1) * w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        for seg in range(n_seg):
            start = n_inc - (seg + 1) * w
            if start < 0:
                break
            segment = profile[start:start + w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        if count > 0:
            f_n = np.sqrt(f2_sum / count)
            if f_n > 1e-15:
                fluctuations.append((w, f_n))

    if len(fluctuations) < 4:
        return 0.5

    log_n = np.log([f[0] for f in fluctuations])
    log_f = np.log([f[1] for f in fluctuations])

    try:
        slope, _, r_value, _, _ = stats.linregress(log_n, log_f)
        if r_value ** 2 < 0.70:
            return 0.5
        return round(max(0.01, min(0.99, slope)), 4)
    except Exception:
        return 0.5


# =============================================================================
# ROLLING Z-SCORE
# =============================================================================

def calculate_rolling_zscore(spread, window=30):
    """Rolling Z-score без lookahead bias. LEGACY — используйте adaptive версию."""
    spread = np.array(spread, dtype=float)
    n = len(spread)
    if n < window + 1:
        mean, std = np.mean(spread), np.std(spread)
        if std < 1e-10:
            return 0.0, np.zeros(n)
        zs = (spread - mean) / std
        return float(zs[-1]), zs

    zscore_series = np.full(n, np.nan)
    for i in range(window, n):
        lb = spread[i - window:i]
        m, s = np.mean(lb), np.std(lb)
        zscore_series[i] = (spread[i] - m) / s if s > 1e-10 else 0.0

    cz = zscore_series[-1]
    return float(0.0 if np.isnan(cz) else cz), zscore_series


def calculate_adaptive_robust_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    """
    Адаптивный робастный Z-score.

    Два улучшения над calculate_rolling_zscore:
      1. Адаптивное окно: Window = clip(2.5 × HL_bars, min_w, max_w)
         Синхронизирует Z-score с ритмом конкретной пары.
      2. MAD вместо std: устойчив к выбросам (fat tails крипто).
         MAD * 1.4826 ≈ sigma для нормального распределения.

    Args:
        spread: массив спреда
        halflife_bars: HL в барах (не часах!). None → default window.
        min_w: минимальное окно (10 — порог стабильности)
        max_w: максимальное окно (60 — не слишком далеко)

    Returns:
        (current_z, z_series, window_used)
    """
    spread = np.array(spread, dtype=float)
    n = len(spread)

    # 1. Адаптивное окно
    if halflife_bars is not None and 0 < halflife_bars < 500:
        window = int(np.clip(round(2.5 * halflife_bars), min_w, max_w))
    else:
        window = 30  # fallback

    if n < window + 1:
        # Мало данных — простой Z
        med = np.median(spread)
        mad = np.median(np.abs(spread - med)) * 1.4826
        if mad < 1e-10:
            return 0.0, np.zeros(n), window
        zs = (spread - med) / mad
        return float(zs[-1]), zs, window

    # 2. Rolling MAD Z-score
    zscore_series = np.full(n, np.nan)
    for i in range(window, n):
        lb = spread[i - window:i]
        med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826

        if mad < 1e-10:
            # fallback на std если MAD = 0 (стейблкоины)
            s = np.std(lb)
            zscore_series[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0.0
        else:
            zscore_series[i] = (spread[i] - med) / mad

    cz = zscore_series[-1]
    return float(0.0 if np.isnan(cz) else cz), zscore_series, window


def calculate_crossing_density(zscore_series, window=None):
    """
    Частота пересечений нуля Z-score.

    Показывает как часто спред реально переходит через mean.
    Высокая плотность → пара активно mean-reverting.
    Низкая плотность → спред "застрял" на одной стороне.

    Args:
        zscore_series: массив Z-scores
        window: количество последних баров для анализа

    Returns:
        float: плотность (0.0–1.0). 0.05 = 5% баров содержат кроссинг.
    """
    z = np.array(zscore_series, dtype=float)
    z = z[~np.isnan(z)]

    if len(z) < 10:
        return 0.0

    if window is not None and len(z) > window:
        z = z[-window:]

    # Считаем смены знака
    signs = np.sign(z)
    # Убираем нули (на нуле — не считается сменой)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0

    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(signs))


def calculate_rolling_correlation(series1, series2, window=30):
    """
    Rolling корреляция Пирсона между двумя ценовыми рядами.

    НЕ используется как фильтр (коинтегрированные пары могут
    временно раскоррелироваться — это момент входа).
    Показывается в UI как информационный индикатор.

    Returns:
        (current_corr, corr_series)
    """
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    if n < window + 1:
        if n > 5:
            return float(np.corrcoef(s1[:n], s2[:n])[0, 1]), np.array([])
        return 0.0, np.array([])

    s1, s2 = s1[:n], s2[:n]
    corr_series = np.full(n, np.nan)

    for i in range(window, n):
        x = s1[i - window:i]
        y = s2[i - window:i]
        sx, sy = np.std(x), np.std(y)
        if sx > 1e-10 and sy > 1e-10:
            corr_series[i] = np.corrcoef(x, y)[0, 1]

    cc = corr_series[-1]
    return float(0.0 if np.isnan(cc) else cc), corr_series


# =============================================================================
# OU PARAMETERS
# =============================================================================

def calculate_ou_parameters(spread, dt=1.0):
    """OU: dX = θ(μ - X)dt + σdW"""
    try:
        if len(spread) < 20:
            return None
        spread = np.array(spread, dtype=float)
        y, x = np.diff(spread), spread[:-1]
        n = len(x)
        sx, sy = np.sum(x), np.sum(y)
        sxy, sx2 = np.sum(x * y), np.sum(x ** 2)
        denom = n * sx2 - sx ** 2
        if abs(denom) < 1e-10:
            return None
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        theta = max(0.001, min(10.0, -b / dt))
        mu = a / theta if theta > 0 else 0.0
        y_pred = a + b * x
        sigma = np.std(y - y_pred)
        halflife = np.log(2) / theta if theta > 0 else 999.0
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            'theta': float(theta), 'mu': float(mu), 'sigma': float(sigma),
            'halflife_ou': float(halflife), 'r_squared': float(r_sq),
            'equilibrium_time': float(-np.log(0.05) / theta if theta > 0 else 999.0)
        }
    except Exception:
        return None


# =============================================================================
# KALMAN FILTER для адаптивного HEDGE RATIO
# =============================================================================

def kalman_hedge_ratio(series1, series2, delta=1e-4, ve=1e-3):
    """
    Kalman Filter для динамического hedge ratio.

    Модель:
      State:  β_t = [intercept_t, hedge_ratio_t]
      Transition: β_t = β_{t-1} + w_t,  w ~ N(0, Q)
      Observation: price1_t = intercept_t + hedge_ratio_t * price2_t + v_t

    Args:
        series1, series2: ценовые ряды (np.array или pd.Series)
        delta: дисперсия перехода (процесс случайного блуждания для β).
               Маленький delta = гладкий HR, большой = быстрая адаптация.
               Default 1e-4 — хороший баланс для 4h крипто.
        ve: начальная дисперсия наблюдения (measurement noise).

    Returns:
        dict:
            hedge_ratios:  np.array — HR на каждом баре
            intercepts:    np.array — intercept на каждом баре
            spread:        np.array — адаптивный спред
            hr_final:      float — текущий (последний) HR
            intercept_final: float
            hr_std:        float — uncertainty текущего HR
            sqrt_Q:        np.array — серия measurement prediction errors
    """
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    if n < 10:
        return None

    s1, s2 = s1[:n], s2[:n]

    # State: [intercept, hedge_ratio]
    # Начальная оценка через OLS на первых 30 барах
    init_n = min(30, n // 3)
    try:
        X_init = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta_init = np.linalg.lstsq(X_init, s1[:init_n], rcond=None)[0]
    except Exception:
        beta_init = np.array([0.0, 1.0])

    # Kalman state
    beta = beta_init.copy()          # [2,] state estimate
    P = np.eye(2) * 1.0              # [2,2] state covariance
    Q = np.eye(2) * delta            # [2,2] transition noise
    R = ve                            # scalar observation noise

    # Storage
    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    innovations = np.zeros(n)    # Kalman innovations (≈ белый шум)
    trading_spread = np.zeros(n) # Торговый спред для Z-score
    sqrt_Q_series = np.zeros(n)

    for t in range(n):
        # Observation vector: x_t = [1, price2_t]
        x_t = np.array([1.0, s2[t]])

        # Predict
        # beta = beta (random walk)
        P = P + Q

        # Update
        y_hat = x_t @ beta                  # predicted price1
        e_t = s1[t] - y_hat                 # innovation
        S_t = x_t @ P @ x_t + R            # innovation variance
        K_t = P @ x_t / S_t                 # Kalman gain [2,]

        beta = beta + K_t * e_t             # state update
        P = P - np.outer(K_t, x_t) @ P     # covariance update

        # Ensure P stays positive definite
        P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))

        # Store
        intercepts[t] = beta[0]
        hedge_ratios[t] = beta[1]
        innovations[t] = e_t
        # Торговый спред: price1 - HR_t * price2 - intercept_t
        trading_spread[t] = s1[t] - beta[1] * s2[t] - beta[0]
        sqrt_Q_series[t] = np.sqrt(max(S_t, 1e-10))

    return {
        'hedge_ratios': hedge_ratios,
        'intercepts': intercepts,
        'spread': trading_spread,       # ← для Z-score и DFA
        'innovations': innovations,     # ← innovations (≈ белый шум)
        'hr_final': float(hedge_ratios[-1]),
        'intercept_final': float(intercepts[-1]),
        'hr_std': float(np.sqrt(P[1, 1])),
        'sqrt_Q': sqrt_Q_series,
        'P_final': P,
    }


def kalman_select_delta(series1, series2, deltas=None):
    """
    Автоподбор delta по максимизации log-likelihood.

    Перебирает несколько значений delta и выбирает лучший.
    Используется если нет уверенности в default delta=1e-4.

    Returns:
        best_delta, best_result, all_likelihoods
    """
    if deltas is None:
        deltas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))

    best_ll = -np.inf
    best_delta = 1e-4
    best_result = None
    all_ll = {}

    for d in deltas:
        res = kalman_hedge_ratio(s1, s2, delta=d)
        if res is None:
            continue

        # Log-likelihood: sum of log N(e_t; 0, S_t)
        sq = res['sqrt_Q']
        innov = res['innovations']  # innovations, not trading spread

        # Ignore first 30 bars (warmup)
        warmup = min(30, n // 3)
        ll_valid = -0.5 * np.sum(
            np.log(2 * np.pi * sq[warmup:]**2 + 1e-10) +
            innov[warmup:]**2 / (sq[warmup:]**2 + 1e-10)
        )

        all_ll[d] = float(ll_valid)
        if ll_valid > best_ll:
            best_ll = ll_valid
            best_delta = d
            best_result = res

    return best_delta, best_result, all_ll




# =============================================================================
# ADF-ТЕСТ СПРЕДА
# =============================================================================

def adf_test_spread(spread, significance=0.05):
    """ADF тест на стационарность спреда."""
    from statsmodels.tsa.stattools import adfuller
    try:
        spread = np.array(spread, dtype=float)
        if len(spread) < 20:
            return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}
        result = adfuller(spread, autolag='AIC')
        return {
            'adf_stat': float(result[0]), 'adf_pvalue': float(result[1]),
            'is_stationary': result[1] < significance,
            'critical_values': {k: float(v) for k, v in result[4].items()}
        }
    except Exception:
        return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}


# =============================================================================
# FDR-КОРРЕКЦИЯ
# =============================================================================

def apply_fdr_correction(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR. Передавайте ВСЕ p-values!"""
    pvalues = np.array(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]

    adjusted = np.empty(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * n / (i + 1)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result, result <= alpha


# =============================================================================
# COINTEGRATION STABILITY
# =============================================================================

def check_cointegration_stability(series1, series2, window_fraction=0.6):
    """4 подокна: полное, начало, конец, середина."""
    from statsmodels.tsa.stattools import coint
    s1, s2 = np.array(series1, dtype=float), np.array(series2, dtype=float)
    n = min(len(s1), len(s2))
    if n < 30:
        return {'is_stable': False, 'windows_passed': 0, 'total_windows': 0,
                'stability_score': 0.0, 'pvalues': []}
    ws = max(20, int(n * window_fraction))
    mid = (n - ws) // 2
    windows = [(0, n), (0, ws), (n - ws, n), (mid, mid + ws)]
    pvalues, passed = [], 0
    for start, end in windows:
        end = min(end, n)
        if end - start < 20:
            continue
        try:
            _, pval, _ = coint(s1[start:end], s2[start:end])
            pvalues.append(float(pval))
            if pval < 0.05:
                passed += 1
        except Exception:
            pvalues.append(1.0)
    total = len(pvalues)
    return {
        'is_stable': passed >= 3, 'windows_passed': passed,
        'total_windows': total,
        'stability_score': round(passed / total if total > 0 else 0.0, 3),
        'pvalues': pvalues
    }


# =============================================================================
# CONFIDENCE
# =============================================================================

def calculate_confidence(hurst, stability_score, fdr_passed, adf_passed,
                         zscore, hedge_ratio, hurst_is_fallback=False,
                         hr_std=None):
    """
    HIGH / MEDIUM / LOW на основе 7 критериев.
    
    v10.1: добавлен критерий HR uncertainty (hr_std/hr < 0.5)
    """
    checks = 0
    total_checks = 7
    if fdr_passed:
        checks += 1
    if adf_passed:
        checks += 1
    if not hurst_is_fallback and hurst != 0.5 and hurst < 0.48:
        checks += 1
    if stability_score >= 0.75:
        checks += 1
    if 0 < hedge_ratio and 0.1 <= abs(hedge_ratio) <= 10.0:
        checks += 1
    if 1.5 <= abs(zscore) <= 5.0:
        checks += 1
    # v10.1: HR uncertainty — Калман уверен в хедже?
    if hr_std is not None and hedge_ratio > 0:
        hr_unc = hr_std / hedge_ratio
        if hr_unc < 0.5:  # uncertainty < 50%
            checks += 1
    else:
        checks += 1  # нет данных — не штрафуем (OLS fallback)

    if checks >= 6:
        return "HIGH", checks, total_checks
    elif checks >= 4:
        return "MEDIUM", checks, total_checks
    else:
        return "LOW", checks, total_checks


# =============================================================================
# [D-B] QUALITY SCORE — насколько пара надёжна
# =============================================================================

def calculate_quality_score(hurst, ou_params, pvalue_adj, stability_score,
                            hedge_ratio, adf_passed=None,
                            hurst_is_fallback=False,
                            crossing_density=None,
                            n_bars=None,
                            hr_std=None):
    """
    Quality Score (0-100) — оценка ПАРЫ, без привязки к текущему Z.

    Компоненты:
      FDR p-value:    25  — статистическая надёжность коинтеграции
      Stability:      25  — устойчивость во времени
      Hurst (DFA):    20  — подтверждение mean-reversion
      ADF:            15  — независимый тест стационарности
      Hedge ratio:    15  — практичность для торговли
                     ----
                     100

    Модификаторы:
      Crossing density < 0.03 → -10 (спред застрял на одной стороне)
      N < 100 баров → -15 (мало данных, ненадёжная статистика)
      HR uncertainty > 50% → -10 (Калман не уверен в хедже)
    """
    bd = {}

    # FDR (25)
    bd['fdr'] = 25 if pvalue_adj <= 0.01 else 20 if pvalue_adj <= 0.03 else 12 if pvalue_adj <= 0.05 else 0

    # Stability (25)
    bd['stability'] = int(stability_score * 25)

    # Hurst (20)
    if hurst_is_fallback or hurst == 0.5:
        bd['hurst'] = 0
    elif hurst <= 0.30:
        bd['hurst'] = 20
    elif hurst <= 0.40:
        bd['hurst'] = 15
    elif hurst <= 0.48:
        bd['hurst'] = 10
    elif hurst < 0.50:
        bd['hurst'] = 4
    else:
        bd['hurst'] = 0

    # ADF (15)
    bd['adf'] = 15 if adf_passed else 0

    # Hedge ratio (15) — v10.4: steeper penalty for extreme HR
    if hedge_ratio <= 0 or abs(hedge_ratio) > 30:
        bd['hedge_ratio'] = 0
    elif 0.2 <= abs(hedge_ratio) <= 5.0:
        bd['hedge_ratio'] = 15
    elif 0.1 <= abs(hedge_ratio) <= 10.0:
        bd['hedge_ratio'] = 10
    elif 0.05 <= abs(hedge_ratio) <= 30.0:
        bd['hedge_ratio'] = 5  # v10.4: covers 10-30 range
    else:
        bd['hedge_ratio'] = 0

    # Crossing density modifier
    if crossing_density is not None and crossing_density < 0.03:
        bd['crossing_penalty'] = -10
    else:
        bd['crossing_penalty'] = 0

    # Data depth modifier (pairs with N<100 get penalty)
    if n_bars is not None and n_bars < 100:
        bd['data_penalty'] = -15
    else:
        bd['data_penalty'] = 0

    # HR uncertainty modifier (v10.1)
    if hr_std is not None and hedge_ratio > 0 and hr_std / hedge_ratio > 0.5:
        bd['hr_unc_penalty'] = -10
    else:
        bd['hr_unc_penalty'] = 0

    total = max(0, min(100, sum(bd.values())))
    return int(total), bd


# =============================================================================
# [D-B] SIGNAL SCORE — насколько сейчас хороший момент для входа
# =============================================================================

def calculate_signal_score(zscore, ou_params, confidence, quality_score=100):
    """
    Signal Score (0-100) — оценка МОМЕНТА входа.
    
    v8.1: Cap по Quality — высокий Z на мусорной паре ≠ хороший сигнал.
    Финальный S = min(raw_S, quality_score * 1.2)

    Компоненты:
      Z-score сила:       40
      Скорость возврата:  30
      Confidence бонус:   30
                         ----
                         100
    """
    bd = {}

    az = abs(zscore)
    if az > 5.0:
        bd['zscore'] = 10
    elif az >= 3.0:
        bd['zscore'] = 40
    elif az >= 2.5:
        bd['zscore'] = 35
    elif az >= 2.0:
        bd['zscore'] = 30
    elif az >= 1.5:
        bd['zscore'] = 20
    elif az >= 1.0:
        bd['zscore'] = 10
    else:
        bd['zscore'] = 0

    if ou_params is not None:
        hl = ou_params['halflife_ou'] * 24
        bd['ou_speed'] = 30 if hl <= 12 else 25 if hl <= 20 else 15 if hl <= 28 else 8 if hl <= 48 else 0
    else:
        bd['ou_speed'] = 0

    if confidence == "HIGH":
        bd['confidence'] = 30
    elif confidence == "MEDIUM":
        bd['confidence'] = 15
    else:
        bd['confidence'] = 0

    raw = max(0, min(100, sum(bd.values())))
    
    # Cap: Signal не может сильно превышать Quality
    cap = int(quality_score * 1.2)
    total = min(raw, cap)
    bd['_cap'] = cap  # Для отладки
    
    return int(total), bd


# =============================================================================
# [v8.1] SANITIZER — жёсткие фильтры-исключения
# =============================================================================

def sanitize_pair(hedge_ratio, stability_passed, stability_total, zscore,
                  n_bars=None, hr_std=None):
    """
    Жёсткий фильтр: пара исключается полностью если не проходит.

    Исключения:
      HR <= 0:        не арбитраж
      |HR| < 0.001:   экономически бессмысленный HR
      |HR| > 30:      фактически односторонняя ставка (v10.4: was 50)
      Stab 0/N:       коинтеграция не подтверждена ни в одном окне
      |Z| > 10:       сломанная модель
      N < 50:         слишком мало данных для надёжной статистики
      HR uncertainty > 100%: Калман не уверен в связи

    Returns:
        (passed, reason)
    """
    if hedge_ratio <= 0:
        return False, f"HR={hedge_ratio:.4f} ≤ 0"
    if abs(hedge_ratio) < 0.001:
        return False, f"|HR|={abs(hedge_ratio):.6f} < 0.001"
    if abs(hedge_ratio) > 30:
        return False, f"|HR|={abs(hedge_ratio):.1f} > 30 (v10.4)"
    if stability_total > 0 and stability_passed == 0:
        return False, f"Stab=0/{stability_total}"
    if abs(zscore) > 10:
        return False, f"|Z|={abs(zscore):.1f} > 10"
    # v10.2: Hard minimum bars
    if n_bars is not None and n_bars < 50:
        return False, f"N={n_bars} < 50 баров"
    # v10.1: HR uncertainty > 100% — Калман не нашёл стабильную связь
    if hr_std is not None and hr_std > 0 and hedge_ratio > 0:
        hr_unc = hr_std / hedge_ratio
        if hr_unc > 1.0:
            return False, f"HR uncertainty {hr_unc:.0%} > 100%"
    return True, "OK"


# =============================================================================
# [v8.1] ADAPTIVE SIGNAL — TF-aware
# =============================================================================

def get_adaptive_signal(zscore, confidence, quality_score, timeframe='4h',
                        stability_ratio=1.0, fdr_passed=True):
    """
    Адаптивный торговый сигнал с учётом таймфрейма.

    v10.4: Hard guards (каскад):
      |Z| > 4.5 AND (Q<70 OR Stab<3/4) → NEUTRAL (структурный слом)
      |Z| > 4.5 AND Q≥70 AND Stab≥3/4  → SIGNAL (extreme, но надёжная пара)
      Q < 30       → NEUTRAL (мусорная пара)
      Q < 40       → max READY (недостаточное качество для SIGNAL)
      Stab < 2/4   → max WATCH (коинтеграция не подтверждена в истории)
      FDR fail + Q<60 → max READY (статистически ненадёжно)

    v8.1 TF-dependent thresholds:
      1h (шумный):  HIGH→2.0, MEDIUM→2.5, LOW→3.0
      4h (базовый): HIGH→1.5, MEDIUM→2.0, LOW→2.5
      1d (дневной): HIGH→1.5, MEDIUM→2.0, LOW→2.5 + min Q≥50

    Args:
        stability_ratio: Доля окон где коинтеграция подтверждена (0.0–1.0).
                         0.25 = 1/4, 0.5 = 2/4, 0.75 = 3/4, 1.0 = 4/4.
        fdr_passed: Прошёл ли FDR (Benjamini-Hochberg) коррекцию.

    Returns:
        (state, direction, threshold_used)
    """
    az = abs(zscore)
    direction = "LONG" if zscore < 0 else "SHORT" if zscore > 0 else "NONE"

    # v10.4: Conditional Z-cap
    if az > 4.5:
        # Только для надёжных пар (Q≥70 + Stab≥3/4) — позволяем extreme signal
        if quality_score >= 70 and stability_ratio >= 0.75:
            pass  # Пропускаем дальше к порогам — это потенциальный V-разворот
        else:
            return "NEUTRAL", "NONE", 4.5

    if quality_score < 30:
        return "NEUTRAL", "NONE", 99.0

    # TF-зависимые пороги
    if timeframe == '1h':
        if confidence == "HIGH" and quality_score >= 50:
            t_signal, t_ready, t_watch = 2.0, 1.5, 1.0
        elif confidence == "MEDIUM" and quality_score >= 40:
            t_signal, t_ready, t_watch = 2.5, 2.0, 1.5
        else:
            t_signal, t_ready, t_watch = 3.0, 2.5, 2.0
    elif timeframe == '1d':
        if confidence == "HIGH" and quality_score >= 50:
            t_signal, t_ready, t_watch = 1.5, 1.2, 0.8
        elif confidence == "MEDIUM" and quality_score >= 50:
            t_signal, t_ready, t_watch = 2.0, 1.5, 1.0
        else:
            t_signal, t_ready, t_watch = 2.5, 2.0, 1.5
    else:
        # 4h — базовый
        if confidence == "HIGH" and quality_score >= 50:
            t_signal, t_ready, t_watch = 1.5, 1.2, 0.8
        elif confidence == "MEDIUM" and quality_score >= 40:
            t_signal, t_ready, t_watch = 2.0, 1.5, 1.0
        else:
            t_signal, t_ready, t_watch = 2.5, 2.0, 1.5

    if az >= t_signal:
        # v10.3: Stability gate — Stab < 2/4 → max WATCH
        if stability_ratio < 0.5:
            return "WATCH", direction, t_signal
        # v10.2: Quality gate — Q < 40 → max READY
        if quality_score < 40:
            return "READY", direction, t_signal
        # v10.4: FDR gate — FDR fail + Q<60 → max READY (ненадёжная статистика)
        if not fdr_passed and quality_score < 60:
            return "READY", direction, t_signal
        return "SIGNAL", direction, t_signal
    elif az >= t_ready:
        # v10.3: Stab < 2/4 → downgrade READY to WATCH
        if stability_ratio < 0.5:
            return "WATCH", direction, t_signal
        return "READY", direction, t_signal
    elif az >= t_watch:
        return "WATCH", direction, t_signal
    else:
        return "NEUTRAL", "NONE", t_signal


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def calculate_trade_score(hurst, ou_params, pvalue_adj, zscore,
                          stability_score, hedge_ratio,
                          adf_passed=None, hurst_is_fallback=False):
    """Legacy — вызывает quality + signal и объединяет."""
    q, qbd = calculate_quality_score(hurst, ou_params, pvalue_adj,
                                      stability_score, hedge_ratio,
                                      adf_passed, hurst_is_fallback)
    # Простое среднее для обратной совместимости
    return q, qbd


def calculate_ou_score(ou_params, hurst):
    """Legacy OU Score."""
    if ou_params is None:
        return 0
    score = 0
    if 0.30 <= hurst <= 0.48: score += 50
    elif 0.48 < hurst <= 0.52: score += 30
    elif 0.25 <= hurst < 0.30: score += 40
    elif hurst < 0.25: score += 25
    elif 0.52 < hurst <= 0.60: score += 15
    hl = ou_params['halflife_ou'] * 24
    if 4 <= hl <= 24: score += 30
    elif 24 < hl <= 48: score += 20
    elif 2 <= hl < 4: score += 15
    elif hl < 2: score += 5
    if ou_params['r_squared'] > 0.15: score += 20
    elif ou_params['r_squared'] > 0.08: score += 15
    elif ou_params['r_squared'] > 0.05: score += 10
    return int(min(100, max(0, score)))


def estimate_exit_time(current_z, theta, mu=0.0, target_z=0.5):
    if theta <= 0.001:
        return 999.0
    try:
        ratio = abs(target_z - mu) / abs(current_z - mu)
        ratio = max(0.001, min(0.999, ratio))
        return -np.log(ratio) / theta
    except Exception:
        return 999.0


def validate_ou_quality(ou_params, hurst=None, min_theta=0.1, max_halflife=100):
    if ou_params is None:
        return False, "No OU"
    if ou_params['theta'] < min_theta:
        return False, "Low theta"
    if ou_params['halflife_ou'] * 24 > max_halflife:
        return False, "High HL"
    if hurst is not None and hurst > 0.70:
        return False, "High Hurst"
    return True, "OK"


# =============================================================================
# ТЕСТ
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  v10.0.0 — Adaptive Robust Z + Crossing Density + Correlation")
    print("=" * 65)
    np.random.seed(42)

    # Generate synthetic mean-reverting spread
    n = 300
    spread_mr = [0.0]
    for i in range(n - 1):
        dx = 0.3 * (0 - spread_mr[-1]) + 0.5 * np.random.randn()
        spread_mr.append(spread_mr[-1] + dx)
    spread_mr = np.array(spread_mr)

    # OU params for HL
    ou = calculate_ou_parameters(spread_mr, dt=1/6)  # 4h
    hl_bars = ou['halflife_ou'] / (4.0) if ou else 10  # HL_hours / hours_per_bar

    # 1. Adaptive Robust Z-score
    print(f"\n--- Adaptive Robust Z-Score ---")
    print(f"OU HL = {ou['halflife_ou']:.1f}ч → {hl_bars:.1f} bars (4h TF)")

    z_old, zs_old = calculate_rolling_zscore(spread_mr, window=30)
    z_new, zs_new, w_used = calculate_adaptive_robust_zscore(spread_mr, halflife_bars=hl_bars)
    print(f"Old (window=30, std):  Z={z_old:+.3f}")
    print(f"New (window={w_used}, MAD): Z={z_new:+.3f}")

    # Test with different HL values
    print(f"\nAdaptive windows for different HL:")
    for hl in [2, 5, 10, 20, 50]:
        _, _, w = calculate_adaptive_robust_zscore(spread_mr, halflife_bars=hl)
        print(f"  HL={hl:>2d} bars → window={w}")

    # 2. Crossing Density
    print(f"\n--- Crossing Density ---")
    cd_mr = calculate_crossing_density(zs_new)
    # Generate "stuck" spread
    stuck = np.concatenate([np.ones(100) * 2, np.ones(100) * -1, np.ones(100) * 1.5])
    cd_stuck = calculate_crossing_density(stuck)
    print(f"Mean-reverting: density={cd_mr:.3f} ({'✅ active' if cd_mr > 0.03 else '❌ stuck'})")
    print(f"Stuck spread:   density={cd_stuck:.3f} ({'✅ active' if cd_stuck > 0.03 else '❌ stuck'})")

    # 3. Rolling Correlation
    print(f"\n--- Rolling Correlation ---")
    s2 = np.cumsum(np.random.randn(n) * 0.3) + 100
    s1 = 1.5 * s2 + np.random.randn(n) * 0.5 + 20
    corr, corr_s = calculate_rolling_correlation(s1, s2, window=30)
    print(f"Correlated pair: ρ={corr:.3f}")
    s1_uncorr = np.cumsum(np.random.randn(n) * 0.3) + 50
    corr_u, _ = calculate_rolling_correlation(s1_uncorr, s2, window=30)
    print(f"Uncorrelated:    ρ={corr_u:.3f}")

    # 4. Sanitizer with min_bars + HR uncertainty
    print(f"\n--- Sanitizer v10.1 ---")
    tests = [
        (1.2, 3, 4, 2.0, 300, 0.1, "normal 300 bars"),
        (1.2, 3, 4, 2.0, 29,  0.1, "only 29 bars"),
        (1.2, 3, 4, 2.0, 30,  0.1, "30 bars (boundary)"),
        (0.0003, 2, 4, -2.3, 300, 0.0, "HR<0.001"),
        (0.18, 3, 4, -1.8, 300, 0.25, "HR unc=139%"),
        (0.18, 3, 4, -1.8, 300, 0.05, "HR unc=28%"),
    ]
    for hr, sp, st, z, nb, hs, name in tests:
        ok, reason = sanitize_pair(hr, sp, st, z, n_bars=nb, hr_std=hs)
        print(f"  {name:<22s} → {'✅' if ok else '❌'} {reason}")

    # 5. Quality with crossing penalty + HR unc penalty
    print(f"\n--- Quality Score v10.1 ---")
    q1, bd1 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.08, hr_std=0.1)
    q2, bd2 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.01, hr_std=0.1)
    q3, bd3 = calculate_quality_score(0.2, ou, 0.01, 0.75, 1.5, True, crossing_density=0.08, hr_std=1.0)
    print(f"Active/good HR: Q={q1} cross={bd1.get('crossing_penalty',0)} hr_unc={bd1.get('hr_unc_penalty',0)}")
    print(f"Stuck/good HR:  Q={q2} cross={bd2.get('crossing_penalty',0)} hr_unc={bd2.get('hr_unc_penalty',0)}")
    print(f"Active/bad HR:  Q={q3} cross={bd3.get('crossing_penalty',0)} hr_unc={bd3.get('hr_unc_penalty',0)}")

    # 6. Signal gates: Q, Z, Stability
    print(f"\n--- Signal Gates v10.3 ---")
    s1, d1, t1 = get_adaptive_signal(2.69, 'LOW', 8, '4h')
    s2, d2, t2 = get_adaptive_signal(2.69, 'LOW', 25, '4h')
    s3, d3, t3 = get_adaptive_signal(-1.83, 'HIGH', 63, '4h')
    s4, d4, t4 = get_adaptive_signal(2.5, 'MEDIUM', 56, '4h', stability_ratio=0.25)
    s5, d5, t5 = get_adaptive_signal(2.5, 'MEDIUM', 56, '4h', stability_ratio=0.75)
    s6, d6, t6 = get_adaptive_signal(4.8, 'LOW', 50, '4h')
    print(f"Q=8  LOW:           {s1} (expect NEUTRAL)")
    print(f"Q=25 LOW:           {s2} (expect NEUTRAL)")
    print(f"Q=63 HIGH:          {s3} (expect SIGNAL)")
    print(f"Q=56 Stab=1/4:      {s4} (expect WATCH)")
    print(f"Q=56 Stab=3/4:      {s5} (expect SIGNAL)")
    print(f"Z=4.8:              {s6} (expect NEUTRAL)")

    print(f"\n✅ v10.3.0 ready!")
