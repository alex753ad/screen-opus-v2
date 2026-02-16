"""
Модуль расчета Hurst Exponent и Ornstein-Uhlenbeck параметров
ВЕРСИЯ v7.0.0: ADF + Fixed FDR + Edge Cases + Confidence

Дата: 16 февраля 2026

ИЗМЕНЕНИЯ v7.0.0:
  [A] Hurst через DFA на инкрементах (из v6.0)
  [B] Rolling Z-score без lookahead bias (из v6.0)
  [C] FDR — теперь принимает ВСЕ p-values (ИСПРАВЛЕНО)
  [D] Rolling cointegration stability (из v6.0)
  [NEW] ADF-тест спреда — независимое подтверждение стационарности
  [NEW] Edge cases: Hurst=0.5 fallback → 0 pts, |Z|>5 → cap, HR<0 → 0 pts
  [NEW] Перевзвешивание: Stability 10→20, Z-score 25→20
  [NEW] Confidence level (LOW/MEDIUM/HIGH)
"""

import numpy as np
from scipy import stats


# =============================================================================
# [A] HURST EXPONENT — DFA
# =============================================================================

def calculate_hurst_exponent(time_series, min_window=4):
    """
    DFA на инкрементах. H < 0.5 mean-reverting, ≈ 0.5 random walk, > 0.5 trending.
    Возвращает 0.5 как fallback при недостатке данных или плохом фите (R² < 0.70).
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
            trend = np.polyval(coeffs, x)
            f2_sum += np.mean((segment - trend) ** 2)
            count += 1
        for seg in range(n_seg):
            start = n_inc - (seg + 1) * w
            if start < 0:
                break
            segment = profile[start:start + w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            f2_sum += np.mean((segment - trend) ** 2)
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
# [B] ROLLING Z-SCORE
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
# [NEW] ADF-ТЕСТ СПРЕДА
# =============================================================================

def adf_test_spread(spread, significance=0.05):
    """
    Augmented Dickey-Fuller тест на стационарность спреда.
    Независимое подтверждение: если спред стационарен → mean-reverting.

    Returns:
        dict: adf_stat, adf_pvalue, is_stationary, critical_values
    """
    from statsmodels.tsa.stattools import adfuller

    try:
        spread = np.array(spread, dtype=float)
        if len(spread) < 20:
            return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}

        result = adfuller(spread, autolag='AIC')
        return {
            'adf_stat': float(result[0]),
            'adf_pvalue': float(result[1]),
            'is_stationary': result[1] < significance,
            'critical_values': {k: float(v) for k, v in result[4].items()}
        }
    except Exception:
        return {'adf_stat': 0, 'adf_pvalue': 1.0, 'is_stationary': False, 'critical_values': {}}


# =============================================================================
# [C] FDR-КОРРЕКЦИЯ (ИСПРАВЛЕННАЯ v7.0)
# =============================================================================

def apply_fdr_correction(pvalues, alpha=0.05):
    """
    Benjamini-Hochberg FDR.

    ВАЖНО v7.0: Передавайте ВСЕ p-values (включая > 0.05)!
    Тогда BH корректно учитывает полное число тестов.

    Args:
        pvalues: список ВСЕХ p-values от коинтеграционных тестов
        alpha: целевой FDR
    Returns:
        (adjusted_pvalues, rejected): массивы той же длины
    """
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
# [D] ROLLING COINTEGRATION STABILITY
# =============================================================================

def check_cointegration_stability(series1, series2, window_fraction=0.6):
    """Проверка на 4 подокнах: полное, начало, конец, середина."""
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
        'is_stable': passed >= 3,
        'windows_passed': passed,
        'total_windows': total,
        'stability_score': round(passed / total if total > 0 else 0.0, 3),
        'pvalues': pvalues
    }


# =============================================================================
# [NEW] КОМПОЗИТНЫЙ TRADE SCORE v7 (перевзвешен + edge cases)
# =============================================================================

def calculate_trade_score(hurst, ou_params, pvalue_adj, zscore,
                          stability_score, hedge_ratio,
                          adf_passed=None, hurst_is_fallback=False):
    """
    Trade Score v7 (0-100).

    ИЗМЕНЕНИЯ v7:
      - Stability: 10 → 20 (ключевой предиктор)
      - Z-score: 25 → 15 (триггер, но не главное)
      - ADF: новый компонент (10)
      - |Z| > 5 → аномалия, cap at 5 pts
      - Hurst = 0.5 fallback → 0 pts
      - HR < 0 → 0 pts (не арбитраж)

    Веса:
      P-value (FDR):   20
      Hurst (DFA):     15
      Stability:       20
      Z-score:         15
      OU half-life:    10
      ADF:             10
      Hedge ratio:     10
                      ----
                      100
    """
    bd = {}

    # --- P-value после FDR (20) ---
    bd['pvalue'] = 20 if pvalue_adj <= 0.01 else 15 if pvalue_adj <= 0.03 else 10 if pvalue_adj <= 0.05 else 0

    # --- Hurst (15) --- EDGE CASE: fallback=0.5 → 0 pts
    if hurst_is_fallback or hurst == 0.5:
        bd['hurst'] = 0  # Не знаем → не даём баллы
    elif hurst <= 0.30:
        bd['hurst'] = 15
    elif hurst <= 0.40:
        bd['hurst'] = 12
    elif hurst <= 0.48:
        bd['hurst'] = 8
    elif hurst < 0.50:
        bd['hurst'] = 4
    else:
        bd['hurst'] = 0  # trending

    # --- Stability (20) ---
    bd['stability'] = int(stability_score * 20)

    # --- Z-score (15) --- EDGE CASE: |Z|>5 → аномалия
    az = abs(zscore)
    if az > 5.0:
        bd['zscore'] = 5  # Аномалия: может быть сломанная модель
    elif az >= 2.5:
        bd['zscore'] = 15
    elif az >= 2.0:
        bd['zscore'] = 12
    elif az >= 1.5:
        bd['zscore'] = 6
    elif az >= 1.0:
        bd['zscore'] = 3
    else:
        bd['zscore'] = 0

    # --- OU half-life (10) ---
    if ou_params is not None:
        hl = ou_params['halflife_ou'] * 24
        bd['halflife'] = 10 if 4 <= hl <= 24 else 7 if hl <= 48 else 5 if 2 <= hl < 4 else 2 if hl < 2 else 0
    else:
        bd['halflife'] = 0

    # --- ADF (10) ---
    if adf_passed is True:
        bd['adf'] = 10
    elif adf_passed is False:
        bd['adf'] = 0
    else:
        bd['adf'] = 0  # Не проводился

    # --- Hedge ratio (10) --- EDGE CASE: HR < 0 → 0
    if hedge_ratio < 0:
        bd['hedge_ratio'] = 0  # Не арбитраж — обе ноги в одну сторону
    else:
        ahr = abs(hedge_ratio)
        bd['hedge_ratio'] = 10 if 0.2 <= ahr <= 5.0 else 7 if 0.1 <= ahr <= 10.0 else 4 if 0.05 <= ahr <= 20.0 else 1

    total = max(0, min(100, sum(bd.values())))
    return int(total), bd


# =============================================================================
# [NEW] CONFIDENCE LEVEL
# =============================================================================

def calculate_confidence(hurst, stability_score, fdr_passed, adf_passed,
                         zscore, hedge_ratio, hurst_is_fallback=False):
    """
    Confidence: LOW / MEDIUM / HIGH.

    Основан на согласованности метрик, а не на абсолютных значениях.
    Трейдеру нужно быстро видеть: можно ли доверять этому сигналу?

    HIGH (все подтверждают):
      - FDR passed
      - ADF passed
      - Hurst < 0.48 (и не fallback)
      - Stability >= 3/4
      - HR > 0 и в разумном диапазоне
      - |Z| в диапазоне 2-5

    MEDIUM (частичное подтверждение):
      - Хотя бы 3 из 6 критериев

    LOW (мало подтверждений):
      - Менее 3 критериев
    """
    checks = 0
    total_checks = 6

    # 1. FDR
    if fdr_passed:
        checks += 1

    # 2. ADF
    if adf_passed:
        checks += 1

    # 3. Hurst
    if not hurst_is_fallback and hurst != 0.5 and hurst < 0.48:
        checks += 1

    # 4. Stability
    if stability_score >= 0.75:  # 3/4 или 4/4
        checks += 1

    # 5. Hedge ratio
    if hedge_ratio > 0 and 0.1 <= abs(hedge_ratio) <= 10.0:
        checks += 1

    # 6. Z-score в нормальном диапазоне
    if 1.5 <= abs(zscore) <= 5.0:
        checks += 1

    if checks >= 5:
        return "HIGH", checks, total_checks
    elif checks >= 3:
        return "MEDIUM", checks, total_checks
    else:
        return "LOW", checks, total_checks


# =============================================================================
# LEGACY
# =============================================================================

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
    print("  v7.0.0 — ADF + Fixed FDR + Edge Cases + Confidence")
    print("=" * 60)

    np.random.seed(42)

    # DFA
    print("\n--- DFA Hurst ---")
    spread_mr = [0.0]
    for i in range(250):
        dx = 0.8 * (0 - spread_mr[-1]) + 0.5 * np.random.randn()
        spread_mr.append(spread_mr[-1] + dx)
    h_mr = calculate_hurst_exponent(spread_mr)
    print(f"Mean-reverting: H={h_mr:.4f}")
    print(f"Random walk:    H={calculate_hurst_exponent(list(np.cumsum(np.random.randn(250)))):.4f}")

    # ADF
    print("\n--- ADF Test ---")
    adf = adf_test_spread(spread_mr)
    print(f"MR spread: stat={adf['adf_stat']:.3f}, p={adf['adf_pvalue']:.4f}, stationary={adf['is_stationary']}")
    rw = np.cumsum(np.random.randn(250))
    adf_rw = adf_test_spread(rw)
    print(f"RW spread: stat={adf_rw['adf_stat']:.3f}, p={adf_rw['adf_pvalue']:.4f}, stationary={adf_rw['is_stationary']}")

    # FDR — передаём ВСЕ p-values (имитация: 7 значимых + 93 нет)
    print("\n--- FDR (all p-values, N=100) ---")
    pvals = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.049] + [0.5 + 0.005*i for i in range(93)]
    adj, rej = apply_fdr_correction(pvals, alpha=0.05)
    for p, a, r in list(zip(pvals[:7], adj[:7], rej[:7])):
        print(f"  p={p:.3f} → adj={a:.4f} {'✅' if r else '❌'}")

    # Trade Score edge cases
    print("\n--- Trade Score Edge Cases ---")
    ou = calculate_ou_parameters(spread_mr, dt=1/6)

    # Normal case
    sc, bd = calculate_trade_score(0.30, ou, 0.01, -2.5, 0.75, 1.2, adf_passed=True)
    print(f"Normal:         {sc}/100  {bd}")

    # Hurst fallback
    sc, bd = calculate_trade_score(0.50, ou, 0.01, -2.5, 0.75, 1.2, adf_passed=True, hurst_is_fallback=True)
    print(f"Hurst fallback: {sc}/100  {bd}")

    # Z > 5 anomaly
    sc, bd = calculate_trade_score(0.30, ou, 0.01, 11.5, 0.25, 45.0, adf_passed=True)
    print(f"|Z|=11.5:       {sc}/100  {bd}")

    # HR < 0
    sc, bd = calculate_trade_score(0.30, ou, 0.01, -2.5, 1.0, -0.13, adf_passed=True)
    print(f"HR<0:           {sc}/100  {bd}")

    # Confidence
    print("\n--- Confidence ---")
    conf, checks, total = calculate_confidence(0.30, 1.0, True, True, -2.5, 1.2)
    print(f"Best case:  {conf} ({checks}/{total})")
    conf, checks, total = calculate_confidence(0.50, 0.25, True, False, 11.5, -0.13, hurst_is_fallback=True)
    print(f"Worst case: {conf} ({checks}/{total})")

    print("\n✅ v7.0.0 ready!")
