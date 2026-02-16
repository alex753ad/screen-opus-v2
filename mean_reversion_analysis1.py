"""
Модуль расчета Hurst Exponent и Ornstein-Uhlenbeck параметров
ВЕРСИЯ v8.0.0: Quality/Signal Score + Adaptive Thresholds + HR Cutoff

Дата: 16 февраля 2026

ИЗМЕНЕНИЯ v8.0.0:
  [D-A] Adaptive Z-threshold по Confidence (HIGH→1.5, MEDIUM→2.0)
  [D-A] Сигналы: SIGNAL/READY/WATCH/NEUTRAL
  [D-B] Раздельные Quality Score и Signal Score
  [D-A] HR > 100 → жёсткий cutoff
  Всё из v7: DFA, ADF, FDR, Stability, Confidence, edge cases
"""

import numpy as np
from scipy import stats


# =============================================================================
# HURST — DFA
# =============================================================================

def calculate_hurst_exponent(time_series, min_window=4):
    """DFA на инкрементах. Возвращает 0.5 при fallback."""
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
    """Rolling Z-score без lookahead bias."""
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
                         zscore, hedge_ratio, hurst_is_fallback=False):
    """HIGH / MEDIUM / LOW на основе 6 критериев."""
    checks = 0
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

    if checks >= 5:
        return "HIGH", checks, 6
    elif checks >= 3:
        return "MEDIUM", checks, 6
    else:
        return "LOW", checks, 6


# =============================================================================
# [D-B] QUALITY SCORE — насколько пара надёжна
# =============================================================================

def calculate_quality_score(hurst, ou_params, pvalue_adj, stability_score,
                            hedge_ratio, adf_passed=None,
                            hurst_is_fallback=False):
    """
    Quality Score (0-100) — оценка ПАРЫ, без привязки к текущему Z.

    Это "стационарная" метрика: если пара качественная, она будет
    качественной и завтра. Используется для Watchlist.

    Компоненты:
      FDR p-value:    25  — статистическая надёжность коинтеграции
      Stability:      25  — устойчивость во времени
      Hurst (DFA):    20  — подтверждение mean-reversion
      ADF:            15  — независимый тест стационарности
      Hedge ratio:    15  — практичность для торговли
                     ----
                     100
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

    # Hedge ratio (15)
    if hedge_ratio <= 0 or abs(hedge_ratio) > 100:
        bd['hedge_ratio'] = 0
    elif 0.2 <= abs(hedge_ratio) <= 5.0:
        bd['hedge_ratio'] = 15
    elif 0.1 <= abs(hedge_ratio) <= 10.0:
        bd['hedge_ratio'] = 10
    elif 0.05 <= abs(hedge_ratio) <= 20.0:
        bd['hedge_ratio'] = 5
    else:
        bd['hedge_ratio'] = 2

    total = max(0, min(100, sum(bd.values())))
    return int(total), bd


# =============================================================================
# [D-B] SIGNAL SCORE — насколько сейчас хороший момент для входа
# =============================================================================

def calculate_signal_score(zscore, ou_params, confidence):
    """
    Signal Score (0-100) — оценка МОМЕНТА входа.

    Меняется каждый скан. Высокий Signal = пора входить.

    Компоненты:
      Z-score сила:       40  — главный триггер
      Скорость возврата:  30  — theta / OU half-life
      Confidence бонус:   30  — HIGH усиливает сигнал
                         ----
                         100
    """
    bd = {}

    # Z-score (40)
    az = abs(zscore)
    if az > 5.0:
        bd['zscore'] = 10  # Аномалия — не доверяем
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

    # OU speed (30)
    if ou_params is not None:
        hl = ou_params['halflife_ou'] * 24  # часы
        if hl <= 12:
            bd['ou_speed'] = 30
        elif hl <= 20:
            bd['ou_speed'] = 25
        elif hl <= 28:
            bd['ou_speed'] = 15
        elif hl <= 48:
            bd['ou_speed'] = 8
        else:
            bd['ou_speed'] = 0
    else:
        bd['ou_speed'] = 0

    # Confidence бонус (30)
    if confidence == "HIGH":
        bd['confidence'] = 30
    elif confidence == "MEDIUM":
        bd['confidence'] = 15
    else:
        bd['confidence'] = 0

    total = max(0, min(100, sum(bd.values())))
    return int(total), bd


# =============================================================================
# [D-A] ADAPTIVE SIGNAL — определение торгового состояния
# =============================================================================

def get_adaptive_signal(zscore, confidence, quality_score):
    """
    Адаптивный торговый сигнал на основе Z-score, Confidence и Quality.

    Состояния:
      SIGNAL  — входить сейчас (сильное отклонение у качественной пары)
      READY   — почти готово к входу (можно входить агрессивно)
      WATCH   — мониторить, приближается к порогу
      NEUTRAL — далеко от входа

    Пороги:
      HIGH confidence + Quality >= 50:   SIGNAL при |Z|≥1.5, READY при |Z|≥1.2
      MEDIUM confidence + Quality >= 40: SIGNAL при |Z|≥2.0, READY при |Z|≥1.5
      LOW / low quality:                 SIGNAL при |Z|≥2.5, READY при |Z|≥2.0

    Returns:
        (state, direction, threshold_used)
        state: SIGNAL/READY/WATCH/NEUTRAL
        direction: LONG/SHORT/NONE
        threshold_used: фактический порог для этой пары
    """
    az = abs(zscore)
    direction = "LONG" if zscore < 0 else "SHORT" if zscore > 0 else "NONE"

    # Аномалия
    if az > 5.0:
        return "NEUTRAL", "NONE", 5.0

    # Определяем пороги
    if confidence == "HIGH" and quality_score >= 50:
        t_signal, t_ready, t_watch = 1.5, 1.2, 0.8
    elif confidence == "MEDIUM" and quality_score >= 40:
        t_signal, t_ready, t_watch = 2.0, 1.5, 1.0
    else:
        t_signal, t_ready, t_watch = 2.5, 2.0, 1.5

    if az >= t_signal:
        return "SIGNAL", direction, t_signal
    elif az >= t_ready:
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
    print("=" * 60)
    print("  v8.0.0 — Quality/Signal Score + Adaptive Thresholds")
    print("=" * 60)
    np.random.seed(42)

    # DFA
    spread_mr = [0.0]
    for i in range(250):
        dx = 0.8 * (0 - spread_mr[-1]) + 0.5 * np.random.randn()
        spread_mr.append(spread_mr[-1] + dx)
    h = calculate_hurst_exponent(spread_mr)
    print(f"\nDFA MR: H={h:.4f}")

    ou = calculate_ou_parameters(spread_mr, dt=1/6)

    # Quality Score
    print("\n--- Quality Score ---")
    q, qbd = calculate_quality_score(0.30, ou, 0.01, 0.75, 1.2, True)
    print(f"Ideal pair:   Q={q}/100 {qbd}")
    q, qbd = calculate_quality_score(0.50, ou, 0.10, 0.25, 37544, True, True)
    print(f"Bad pair:     Q={q}/100 {qbd}")

    # Signal Score
    print("\n--- Signal Score ---")
    s, sbd = calculate_signal_score(-2.5, ou, "HIGH")
    print(f"|Z|=2.5 HIGH: S={s}/100 {sbd}")
    s, sbd = calculate_signal_score(-1.5, ou, "HIGH")
    print(f"|Z|=1.5 HIGH: S={s}/100 {sbd}")
    s, sbd = calculate_signal_score(-0.5, ou, "MEDIUM")
    print(f"|Z|=0.5 MED:  S={s}/100 {sbd}")

    # Adaptive Signal
    print("\n--- Adaptive Signal ---")
    for z, conf, q in [(-2.5, "HIGH", 60), (-1.8, "HIGH", 58),
                        (-1.5, "HIGH", 55), (-1.2, "HIGH", 50),
                        (2.1, "MEDIUM", 47), (1.5, "MEDIUM", 44),
                        (11.5, "MEDIUM", 40)]:
        state, dir, thr = get_adaptive_signal(z, conf, q)
        print(f"  Z={z:+.1f} {conf:6s} Q={q}: {state:8s} {dir:5s} (thr={thr})")

    print("\n✅ v8.0.0 ready!")
