#!/usr/bin/env python3
"""
AI Finansrådgiver – Oslo Børs Agent
=====================================
Henter aksjekurser, beregner tekniske indikatorer, genererer kjøps-/salgssignaler,
oppdaterer portefølje og genererer en HTML-dashboard.

Kjør: python agent.py
Krav: pip install yfinance pandas numpy requests
"""

import json
import os
import sys
import math
import hashlib
import datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WATCHLIST_FILE   = BASE_DIR / "watchlist.json"
PORTFOLIO_FILE   = BASE_DIR / "portfolio.json"
RESULTS_FILE     = BASE_DIR / "analysis_results.json"
HISTORY_FILE     = BASE_DIR / "portfolio_history.json"
DASHBOARD_FILE   = BASE_DIR / "dashboard.html"

# ─── Dependency check ─────────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    try: import yfinance
    except ImportError: missing.append("yfinance")
    try: import pandas
    except ImportError: missing.append("pandas")
    try: import numpy
    except ImportError: missing.append("numpy")
    if missing:
        print(f"\n❌ Manglende pakker: {', '.join(missing)}")
        print(f"   Kjør: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import yfinance as yf
import pandas as pd
import numpy as np

# ─── Technical Indicators ─────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close: pd.Series, period=20, num_std=2):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (close - lower) / (upper - lower)   # 0 = lower band, 1 = upper band
    return upper, sma, lower, pct_b


def calc_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_volume_ratio(volume: pd.Series, period=20) -> pd.Series:
    """Current volume / 20-day average volume."""
    return volume / volume.rolling(period).mean()


# ─── Signal Generation ────────────────────────────────────────────────────────

def generate_signal(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Score-based signal engine for swing trading.
    Returns a dict with score, label, reasons, and key indicator values.
    Score range: –85 to +85
      >= 40  → STERKT KJØP
      20–39  → KJØP
      –19–19 → HOLD
      –39––20 → SELG
      <= –40 → STERKT SELG
    """
    if len(df) < 60:
        return {"signal": "UTILSTREKKELIG DATA", "score": 0, "reasons": [], "indicators": {}}

    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]

    # Indicators
    rsi           = calc_rsi(close, cfg.get("rsi_period", 14))
    macd, sig, hist = calc_macd(close, cfg.get("macd_fast", 12),
                                  cfg.get("macd_slow", 26), cfg.get("macd_signal", 9))
    bb_upper, bb_mid, bb_lower, pct_b = calc_bollinger(
        close, cfg.get("bollinger_period", 20), cfg.get("bollinger_std", 2))
    sma20  = calc_sma(close, cfg.get("sma_short", 20))
    sma50  = calc_sma(close, cfg.get("sma_long",  50))
    vol_ratio = calc_volume_ratio(volume)

    # Latest values
    r      = rsi.iloc[-1]
    h      = hist.iloc[-1]
    h_prev = hist.iloc[-2]
    pb     = pct_b.iloc[-1]
    p      = close.iloc[-1]
    p_prev = close.iloc[-2]
    s20    = sma20.iloc[-1]
    s50    = sma50.iloc[-1]
    vr     = vol_ratio.iloc[-1]
    atr    = calc_atr(high, low, close).iloc[-1]
    atr_pct = (atr / p) * 100 if p > 0 else 0

    score   = 0
    reasons = []

    # ── RSI ──────────────────────────────────────────────────
    if not math.isnan(r):
        if r < 30:
            score += 30; reasons.append(f"RSI {r:.0f} – kraftig oversolgt (+30)")
        elif r < 40:
            score += 15; reasons.append(f"RSI {r:.0f} – oversolgt (+15)")
        elif r > 70:
            score -= 30; reasons.append(f"RSI {r:.0f} – kraftig overkjøpt (–30)")
        elif r > 60:
            score -= 15; reasons.append(f"RSI {r:.0f} – overkjøpt (–15)")

    # ── Trend / Moving Averages ───────────────────────────────
    if not (math.isnan(s20) or math.isnan(s50)):
        if p > s20 > s50:
            score += 20; reasons.append(f"Kurs > SMA20 > SMA50 – opptrend (+20)")
        elif p < s20 < s50:
            score -= 20; reasons.append(f"Kurs < SMA20 < SMA50 – nedtrend (–20)")
        elif p > s20:
            score += 8;  reasons.append(f"Kurs over SMA20 (+8)")
        elif p < s20:
            score -= 8;  reasons.append(f"Kurs under SMA20 (–8)")

    # ── SMA20 crossover (signal from yesterday) ───────────────
    if not (math.isnan(sma20.iloc[-2]) or math.isnan(s20)):
        crossed_above = p_prev < sma20.iloc[-2] and p >= s20
        crossed_below = p_prev > sma20.iloc[-2] and p <= s20
        if crossed_above:
            score += 10; reasons.append("Kurs krysset over SMA20 i dag (+10)")
        elif crossed_below:
            score -= 10; reasons.append("Kurs krysset under SMA20 i dag (–10)")

    # ── MACD ─────────────────────────────────────────────────
    if not (math.isnan(h) or math.isnan(h_prev)):
        if h > 0 and h > h_prev:
            score += 20; reasons.append(f"MACD histogram positivt og stigende (+20)")
        elif h > 0:
            score += 8;  reasons.append(f"MACD histogram positivt (+8)")
        elif h < 0 and h < h_prev:
            score -= 20; reasons.append(f"MACD histogram negativt og fallende (–20)")
        elif h < 0:
            score -= 8;  reasons.append(f"MACD histogram negativt (–8)")

    # ── Bollinger Bands ────────────────────────────────────────
    if not math.isnan(pb):
        if pb < 0.05:
            score += 15; reasons.append(f"Kurs nær nedre Bollinger-band ({pb:.2f}) (+15)")
        elif pb > 0.95:
            score -= 15; reasons.append(f"Kurs nær øvre Bollinger-band ({pb:.2f}) (–15)")

    # ── Volume ────────────────────────────────────────────────
    if not math.isnan(vr):
        if vr > 1.5 and p > p_prev:
            score += 10; reasons.append(f"Høyt volum på oppgang ({vr:.1f}x snitt) (+10)")
        elif vr > 1.5 and p < p_prev:
            score -= 10; reasons.append(f"Høyt volum på nedgang ({vr:.1f}x snitt) (–10)")

    # ── Signal label ──────────────────────────────────────────
    thresholds = cfg.get("signal_thresholds", {})
    if score >= thresholds.get("strong_buy", 40):
        label = "STERKT KJØP"
    elif score >= thresholds.get("buy", 20):
        label = "KJØP"
    elif score <= thresholds.get("strong_sell", -40):
        label = "STERKT SELG"
    elif score <= thresholds.get("sell", -20):
        label = "SELG"
    else:
        label = "HOLD"

    # ── Position size suggestion (for <100k NOK) ──────────────
    if label in ("STERKT KJØP", "KJØP"):
        conviction = min(abs(score) / 85, 1.0)
        suggested_pct = round(10 + conviction * 15, 0)   # 10–25% of portfolio
    else:
        suggested_pct = 0

    return {
        "signal":           label,
        "score":            round(score, 1),
        "suggested_pct":    suggested_pct,
        "reasons":          reasons,
        "indicators": {
            "rsi":           round(r,   1) if not math.isnan(r)   else None,
            "macd_hist":     round(h,   4) if not math.isnan(h)   else None,
            "sma20":         round(s20, 2) if not math.isnan(s20) else None,
            "sma50":         round(s50, 2) if not math.isnan(s50) else None,
            "bb_pct_b":      round(pb,  3) if not math.isnan(pb)  else None,
            "vol_ratio":     round(vr,  2) if not math.isnan(vr)  else None,
            "atr_pct":       round(atr_pct, 2),
            "price":         round(p, 2),
        }
    }


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_stock(symbol: str, days: int = 90) -> pd.DataFrame | None:
    try:
        period = f"{days}d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            print(f"   ⚠️  Ingen data for {symbol}")
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        print(f"   ❌ Feil ved henting av {symbol}: {e}")
        return None


def fetch_quote(symbol: str) -> dict:
    """Get latest price info for a single symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return {
            "price":         getattr(info, "last_price",     None),
            "prev_close":    getattr(info, "previous_close", None),
            "market_cap":    getattr(info, "market_cap",     None),
            "52w_high":      getattr(info, "year_high",      None),
            "52w_low":       getattr(info, "year_low",       None),
            "currency":      getattr(info, "currency",       "NOK"),
        }
    except Exception:
        return {}


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch P/E, P/B, dividend yield from Yahoo Finance."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "pe_ratio":       info.get("trailingPE"),
            "pb_ratio":       info.get("priceToBook"),
            "div_yield_pct":  (info.get("dividendYield") or 0) * 100,
            "eps":            info.get("trailingEps"),
            "revenue_growth": info.get("revenueGrowth"),
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
        }
    except Exception:
        return {}


def price_history_to_list(df: pd.DataFrame, n: int = 60) -> list:
    """Convert last n rows of OHLCV to a serialisable list for the dashboard."""
    rows = df.tail(n)
    return [
        {
            "date":   row.Index.strftime("%Y-%m-%d"),
            "open":   round(row.Open,   2),
            "high":   round(row.High,   2),
            "low":    round(row.Low,    2),
            "close":  round(row.Close,  2),
            "volume": int(row.Volume),
        }
        for row in rows.itertuples()
    ]


# ─── Portfolio helpers ────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_portfolio_values(portfolio: dict, quotes: dict) -> dict:
    """Recalculate unrealised P&L and total portfolio value."""
    total_invested = 0
    unrealized_pnl = 0
    for h in portfolio.get("holdings", []):
        sym   = h["symbol"]
        q     = quotes.get(sym, {})
        price = q.get("price") or h.get("entry_price")
        h["current_price"]    = round(price, 2) if price else h.get("entry_price")
        h["current_value"]    = round(h["shares"] * h["current_price"], 2)
        h["unrealized_pnl"]   = round(h["current_value"] - h["cost_basis"], 2)
        h["unrealized_pnl_pct"] = round(
            (h["current_price"] / h["entry_price"] - 1) * 100, 2
        ) if h.get("entry_price") else 0
        total_invested += h["current_value"]
        unrealized_pnl += h["unrealized_pnl"]

    total_value = portfolio["capital"]["cash"] + total_invested
    portfolio["capital"]["invested"]          = round(total_invested, 2)
    portfolio["capital"]["total_value"]       = round(total_value, 2)
    portfolio["performance"]["unrealized_pnl"] = round(unrealized_pnl, 2)
    portfolio["performance"]["total_return_nok"] = round(
        total_value - portfolio["capital"]["initial"], 2
    )
    portfolio["performance"]["total_return_pct"] = round(
        (total_value / portfolio["capital"]["initial"] - 1) * 100, 2
    )
    portfolio["performance"]["last_updated"] = datetime.date.today().isoformat()
    return portfolio


def check_stop_loss_take_profit(portfolio: dict) -> list[str]:
    """Return a list of alert strings for holdings that hit SL/TP."""
    alerts = []
    rules  = portfolio.get("rules", {})
    sl_pct = rules.get("stop_loss_pct",   7)
    tp_pct = rules.get("take_profit_pct", 15)
    for h in portfolio.get("holdings", []):
        pnl = h.get("unrealized_pnl_pct", 0)
        if pnl <= -sl_pct:
            alerts.append(f"⛔ STOPP-LOSS: {h['symbol']} er ned {pnl:.1f}% – vurder salg!")
        elif pnl >= tp_pct:
            alerts.append(f"🎯 TA-PROFITT: {h['symbol']} er opp {pnl:.1f}% – vurder salg!")
    return alerts


def update_portfolio_history(portfolio: dict) -> dict:
    """Append today's portfolio snapshot to portfolio_history.json."""
    history = load_json(HISTORY_FILE)
    if "snapshots" not in history:
        history["snapshots"] = []

    cap  = portfolio.get("capital", {})
    perf = portfolio.get("performance", {})
    today = datetime.date.today().isoformat()

    # Replace today's entry if it exists, otherwise append
    snapshot = {
        "date":           today,
        "total_value":    cap.get("total_value", cap.get("initial", 0)),
        "cash":           cap.get("cash", 0),
        "invested":       cap.get("invested", 0),
        "initial":        cap.get("initial", 0),
        "return_nok":     perf.get("total_return_nok", 0),
        "return_pct":     perf.get("total_return_pct", 0),
        "realized_pnl":   perf.get("realized_pnl", 0),
        "unrealized_pnl": perf.get("unrealized_pnl", 0),
        "n_positions":    len(portfolio.get("holdings", [])),
    }

    snaps = history["snapshots"]
    if snaps and snaps[-1]["date"] == today:
        snaps[-1] = snapshot          # overwrite today
    else:
        snaps.append(snapshot)

    # Keep last 365 days
    history["snapshots"] = snaps[-365:]
    save_json(HISTORY_FILE, history)
    return history


# ─── Investment Sizing ────────────────────────────────────────────────────────

def calculate_suggested_nok(signal_label: str, score: float, portfolio: dict) -> int:
    """
    Return a suggested NOK investment amount for buy signals.
    Scales with conviction (score) and respects portfolio rules.
    """
    if signal_label not in ("STERKT KJØP", "KJØP"):
        return 0

    cap   = portfolio.get("capital", {})
    rules = portfolio.get("rules", {})
    cash  = cap.get("cash", 0)
    total = cap.get("total_value", cap.get("initial", 100_000))

    max_pos_pct  = rules.get("max_single_stock_pct", 25) / 100   # default 25 %
    min_cash_pct = rules.get("min_cash_pct", 20) / 100           # keep 20 % cash

    available = cash - total * min_cash_pct
    if available < 2_000:
        return 0

    # Linear scale: score 20 → 10 % of portfolio, score 85 → 25 %
    conviction = min(max((abs(score) - 20) / 65, 0.0), 1.0)
    target_pct = 0.10 + conviction * 0.15        # 10–25 %

    suggested = min(
        total    * target_pct,
        total    * max_pos_pct,
        available,
    )
    # Round to nearest 500 NOK
    return max(0, round(suggested / 500) * 500)


def generate_rationale(signal_label: str, score: float, reasons: list, rsi) -> str:
    """One-sentence plain-Norwegian explanation of the signal."""
    primary   = reasons[0].split(" –")[0] if reasons else "tekniske indikatorer"
    secondary = reasons[1].split(" –")[0] if len(reasons) > 1 else ""

    if signal_label == "STERKT KJØP":
        txt = f"Sterkt kjøpssignal: {primary.lower()}."
        if secondary:
            txt += f" Støttes av {secondary.lower()}."
    elif signal_label == "KJØP":
        txt = f"Kjøpssignal: {primary.lower()}."
        if secondary:
            txt += f" I tillegg {secondary.lower()}."
    elif signal_label == "STERKT SELG":
        txt = f"Sterkt salgssignal: {primary.lower()}. Vurder å redusere posisjon."
    elif signal_label == "SELG":
        txt = f"Salgssignal (score {score:+.0f}): {primary.lower()}."
    else:
        if rsi and rsi < 45:
            txt = f"Hold – RSI {rsi:.0f} antyder mulig støtte, men ikke nok momentum ennå."
        elif rsi and rsi > 55:
            txt = f"Hold – RSI {rsi:.0f}, noe overkjøpt men ikke klart salgssignal."
        else:
            txt = f"Hold – mixed signaler, ingen klar retning (score {score:+.0f})."
    return txt


# ─── HTML Dashboard Generator ─────────────────────────────────────────────────

def generate_html_dashboard(results: dict, portfolio: dict, history: dict,
                             auth: dict | None = None) -> str:
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    ts  = results.get("generated_at", now)

    # Inject data as JS
    results_js   = json.dumps(results,   ensure_ascii=False)
    portfolio_js = json.dumps(portfolio, ensure_ascii=False)
    history_js   = json.dumps(history.get("snapshots", []), ensure_ascii=False)

    # Auth: compute SHA-256 of "username:password" for browser verification
    auth_user = (auth or {}).get("username", "eyvind")
    auth_pass = (auth or {}).get("password", "oslo2026")
    auth_hash = hashlib.sha256(f"{auth_user}:{auth_pass}".encode()).hexdigest()

    html = f"""<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Finansrådgiver">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#1e3a5f">
<title>AI Finansrådgiver – Oslo Børs</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<style>
:root {{
  --bg:#f0f4f8; --card:#fff; --text:#1a2035; --muted:#6b7280;
  --border:#e5e7eb; --accent:#2563eb; --accent2:#1d4ed8;
  --green:#16a34a; --red:#dc2626; --orange:#ea580c; --yellow:#b45309;
  --radius:12px; --shadow:0 2px 8px rgba(0,0,0,.08);
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);font-size:14px}}
.header{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;
  padding:18px 28px;display:flex;align-items:center;justify-content:space-between}}
.header h1{{font-size:1.4rem;font-weight:700;letter-spacing:-.3px}}
.header .subtitle{{font-size:.8rem;opacity:.8;margin-top:2px}}
.ts{{font-size:.75rem;opacity:.7}}
.tabs{{display:flex;background:#1e3a5f;border-bottom:3px solid var(--accent)}}
.tab{{padding:12px 22px;cursor:pointer;color:rgba(255,255,255,.7);
  font-size:.85rem;font-weight:500;transition:all .2s;white-space:nowrap}}
.tab:hover{{color:#fff;background:rgba(255,255,255,.1)}}
.tab.active{{color:#fff;background:var(--accent);border-bottom:3px solid #fff}}
.page{{display:none;padding:24px;max-width:1400px;margin:0 auto}}
.page.active{{display:block}}
.row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}}
.card{{background:var(--card);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:20px;flex:1;min-width:180px}}
.card h3{{font-size:.75rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;color:var(--muted);margin-bottom:8px}}
.card .value{{font-size:1.7rem;font-weight:700;line-height:1}}
.card .change{{font-size:.85rem;margin-top:4px}}
.pos{{color:var(--green)}} .neg{{color:var(--red)}} .neu{{color:var(--muted)}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:.75rem;font-weight:700;letter-spacing:.3px}}
.badge-SK{{background:#dcfce7;color:#166534}}
.badge-K {{background:#dbeafe;color:#1e40af}}
.badge-H {{background:#f3f4f6;color:#374151}}
.badge-S {{background:#fee2e2;color:#991b1b}}
.badge-SS{{background:#fecaca;color:#7f1d1d}}
.badge-UD{{background:#fef3c7;color:#92400e}}
/* ── Tooltips ── */
.tip{{position:relative;border-bottom:1px dashed var(--muted);cursor:help;white-space:nowrap}}
.tip::after{{content:attr(data-tip);position:absolute;bottom:130%;left:50%;
  transform:translateX(-50%);background:#1e293b;color:#e2e8f0;
  padding:8px 12px;border-radius:8px;font-size:.75rem;font-weight:400;
  line-height:1.45;white-space:normal;width:230px;text-align:left;
  opacity:0;pointer-events:none;transition:opacity .18s;z-index:200;
  box-shadow:0 4px 12px rgba(0,0,0,.25)}}
.tip::before{{content:"";position:absolute;bottom:calc(130% - 6px);left:50%;
  transform:translateX(-50%);border:6px solid transparent;
  border-top-color:#1e293b;opacity:0;transition:opacity .18s;z-index:200}}
.tip:hover::after,.tip:hover::before{{opacity:1}}
/* ── Capital modal ── */
.modal-bg{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:500;
  align-items:center;justify-content:center}}
.modal-bg.open{{display:flex}}
.modal{{background:var(--card);border-radius:16px;padding:28px;width:90%;max-width:420px;
  box-shadow:0 8px 32px rgba(0,0,0,.25)}}
.modal h2{{font-size:1.05rem;font-weight:700;margin-bottom:18px}}
.modal label{{font-size:.78rem;font-weight:600;color:var(--muted);display:block;margin-bottom:4px;margin-top:12px}}
.modal input{{width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:8px;
  font:inherit;font-size:.95rem}}
.modal-btns{{display:flex;gap:10px;margin-top:20px}}
.modal-btns button{{flex:1;padding:10px;border-radius:8px;border:none;font:inherit;
  font-weight:700;cursor:pointer}}
.btn-save{{background:var(--accent);color:#fff}}
.btn-cancel{{background:#f1f5f9;color:var(--text)}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{background:#f8fafc;font-weight:600;color:var(--muted);
  text-transform:uppercase;font-size:.7rem;letter-spacing:.5px;
  padding:10px 12px;text-align:left;border-bottom:2px solid var(--border)}}
td{{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:hover td{{background:#f9fafb}}
.section-title{{font-size:1rem;font-weight:700;margin-bottom:12px;
  color:var(--text);display:flex;align-items:center;gap:8px}}
.chart-wrap{{background:var(--card);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:20px;margin-bottom:16px}}
.alert{{background:#fef3c7;border-left:4px solid #f59e0b;
  padding:12px 16px;border-radius:6px;margin-bottom:8px;font-size:.85rem}}
.alert-danger{{background:#fee2e2;border-color:var(--red)}}
select,button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--border);
  cursor:pointer;background:var(--card)}}
button.primary{{background:var(--accent);color:#fff;border:none;font-weight:600}}
button.primary:hover{{background:var(--accent2)}}
.signal-bar{{height:6px;border-radius:3px;margin-top:8px}}
.form-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.form-grid label{{font-size:.8rem;font-weight:600;color:var(--muted);display:block;margin-bottom:4px}}
.form-grid input{{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:6px;font:inherit}}
.legend-dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}}
.macro-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.macro-card{{background:var(--card);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:16px}}
.macro-card h4{{font-size:.75rem;color:var(--muted);font-weight:600;text-transform:uppercase;margin-bottom:6px}}
.macro-card .val{{font-size:1.4rem;font-weight:700}}
.macro-card .chg{{font-size:.8rem;margin-top:3px}}
.reasons-list{{font-size:.8rem;color:var(--muted);list-style:none;margin-top:8px}}
.reasons-list li::before{{content:"•";margin-right:6px;color:var(--accent)}}
.empty{{text-align:center;padding:40px;color:var(--muted)}}
.indicator-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:12px}}
.ind-card{{background:#f8fafc;border-radius:8px;padding:12px;text-align:center}}
.ind-card .ind-val{{font-size:1.1rem;font-weight:700}}
.ind-card .ind-lbl{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
/* ── Mobile styles ─────────────────────────────── */
@media(max-width:600px){{
  .row{{flex-direction:column}}
  .header{{padding:12px 16px}}
  .header h1{{font-size:1.1rem}}
  .header .subtitle{{display:none}}
  /* Hide top tabs on mobile — use bottom nav instead */
  .tabs{{display:none}}
  .page{{padding:14px 12px 90px}}  /* bottom padding for nav */
  table{{font-size:.75rem}}
  th,td{{padding:7px 8px}}
  /* Scrollable tables */
  .table-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
  .card{{padding:14px}}
  .card .value{{font-size:1.4rem}}
  .chart-wrap{{padding:14px}}
  .form-grid{{grid-template-columns:1fr 1fr}}
  .indicator-grid{{grid-template-columns:repeat(3,1fr)}}
  /* Analyse page: stack stock selector vertically */
  #page-analyse>div:first-child{{flex-direction:column;align-items:stretch}}
  #stock-select{{width:100%;font-size:.9rem}}
  /* Chat panel: full-width on narrow screens */
  #chat-panel{{right:0;left:0;width:100%;border-radius:16px 16px 0 0;bottom:0}}
  #chat-fab{{bottom:72px;right:16px}}
  /* Macro grid: 2 columns on phone */
  .macro-grid{{grid-template-columns:1fr 1fr}}
  /* Portfolio add-form: single column on phone */
  .form-grid{{grid-template-columns:1fr}}
  /* Bottom nav */
  .bottom-nav{{
    display:flex!important;
    position:fixed;bottom:0;left:0;right:0;
    background:#1e3a5f;border-top:1px solid rgba(255,255,255,.15);
    z-index:1000;padding-bottom:env(safe-area-inset-bottom);
    overflow-x:auto;-webkit-overflow-scrolling:touch;
    scrollbar-width:none
  }}
  .bottom-nav::-webkit-scrollbar{{display:none}}
  .bnav-item{{
    flex:0 0 auto;min-width:60px;display:flex;flex-direction:column;align-items:center;
    padding:10px 6px 8px;color:rgba(255,255,255,.6);
    font-size:.58rem;font-weight:600;cursor:pointer;text-transform:uppercase;
    letter-spacing:.3px;gap:3px;border:none;background:none;
  }}
  .bnav-item .icon{{font-size:1.2rem;line-height:1}}
  .bnav-item.active{{color:#fff}}
  .bnav-item.active .icon{{filter:drop-shadow(0 0 4px rgba(255,255,255,.5))}}
}}
@media(min-width:601px){{
  .bottom-nav{{display:none!important}}
}}
</style>
</head>
<body>

<!-- ═══════════════════════════ LOGIN SCREEN ═══════════════════════════ -->
<div id="login-screen" style="display:none;position:fixed;inset:0;background:linear-gradient(135deg,#1e3a5f,#2563eb);
  z-index:9999;align-items:center;justify-content:center">
  <div style="background:#fff;border-radius:20px;padding:36px 32px;width:90%;max-width:380px;
    box-shadow:0 20px 60px rgba(0,0,0,.35)">
    <div style="text-align:center;margin-bottom:24px">
      <div style="font-size:2.5rem;margin-bottom:8px">📈</div>
      <h1 style="font-size:1.3rem;font-weight:700;color:#1e3a5f;margin-bottom:4px">AI Finansrådgiver</h1>
      <p style="font-size:.85rem;color:#6b7280">Oslo Børs · Swing Trading</p>
    </div>
    <div style="margin-bottom:14px">
      <label style="font-size:.8rem;font-weight:600;color:#374151;display:block;margin-bottom:6px">Brukernavn</label>
      <input id="login-user" type="text" autocomplete="username" placeholder="brukernavn"
        style="width:100%;padding:12px 14px;border:2px solid #e5e7eb;border-radius:10px;
        font-size:1rem;outline:none;transition:border .2s"
        onfocus="this.style.borderColor='#2563eb'" onblur="this.style.borderColor='#e5e7eb'"
        onkeydown="if(event.key==='Enter')document.getElementById('login-pass').focus()">
    </div>
    <div style="margin-bottom:20px">
      <label style="font-size:.8rem;font-weight:600;color:#374151;display:block;margin-bottom:6px">Passord</label>
      <input id="login-pass" type="password" autocomplete="current-password" placeholder="passord"
        style="width:100%;padding:12px 14px;border:2px solid #e5e7eb;border-radius:10px;
        font-size:1rem;outline:none;transition:border .2s"
        onfocus="this.style.borderColor='#2563eb'" onblur="this.style.borderColor='#e5e7eb'"
        onkeydown="if(event.key==='Enter')doLogin()">
    </div>
    <div id="login-err" style="color:#dc2626;font-size:.82rem;margin-bottom:12px;display:none;
      background:#fee2e2;padding:8px 12px;border-radius:8px;text-align:center">
      Feil brukernavn eller passord
    </div>
    <button onclick="doLogin()" style="width:100%;padding:13px;background:linear-gradient(135deg,#1e3a5f,#2563eb);
      color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;
      transition:opacity .2s" onmouseover="this.style.opacity='.9'" onmouseout="this.style.opacity='1'">
      Logg inn
    </button>
    <p style="font-size:.72rem;color:#9ca3af;text-align:center;margin-top:16px">
      Sesjon lagres i 7 dager
    </p>
  </div>
</div>

<!-- ═══════════════════════════ MAIN APP (hidden until login) ══════════ -->
<div id="app-root" style="display:none">

<div class="header">
  <div>
    <h1>📈 AI Finansrådgiver – Oslo Børs</h1>
    <div class="subtitle">Swing trading · Moderat risiko · Maks 100 000 NOK</div>
  </div>
  <div class="ts">Oppdatert: <span id="ts">{ts}</span></div>
</div>

<!-- Capital adjustment modal -->
<div class="modal-bg" id="capital-modal">
  <div class="modal">
    <h2>⚙️ Juster kapital</h2>
    <label>Total kapital (NOK) – startbeløp du regner avkastning fra</label>
    <input type="number" id="adj-initial" placeholder="f.eks. 100000">
    <label>Tilgjengelige kontanter nå (NOK)</label>
    <input type="number" id="adj-cash" placeholder="f.eks. 85000">
    <p style="font-size:.75rem;color:var(--muted);margin-top:10px">
      Bruk dette når du tilfører eller tar ut kapital, eller for å korrigere beløpene.
    </p>
    <div class="modal-btns">
      <button class="btn-save" onclick="saveCapital()">Lagre</button>
      <button class="btn-cancel" onclick="closeCapitalModal()">Avbryt</button>
    </div>
  </div>
</div>

<!-- Desktop tab navigation -->
<div class="tabs">
  <div class="tab active" onclick="showTab('oversikt')">📊 Oversikt</div>
  <div class="tab" onclick="showTab('signaler')">🎯 Signaler</div>
  <div class="tab" onclick="showTab('analyse')">📉 Analyse</div>
  <div class="tab" onclick="showTab('portefolje')">💼 Portefølje</div>
  <div class="tab" onclick="showTab('makro')">🌍 Makro</div>
  <div class="tab" onclick="showTab('historikk')">📈 Historikk</div>
</div>

<!-- Mobile bottom navigation -->
<nav class="bottom-nav" style="display:none">
  <button class="bnav-item active" id="bn-oversikt" onclick="showTab('oversikt','bn-oversikt')">
    <span class="icon">📊</span>Oversikt
  </button>
  <button class="bnav-item" id="bn-signaler" onclick="showTab('signaler','bn-signaler')">
    <span class="icon">🎯</span>Signaler
  </button>
  <button class="bnav-item" id="bn-analyse" onclick="showTab('analyse','bn-analyse')">
    <span class="icon">📉</span>Analyse
  </button>
  <button class="bnav-item" id="bn-portefolje" onclick="showTab('portefolje','bn-portefolje')">
    <span class="icon">💼</span>Portefølje
  </button>
  <button class="bnav-item" id="bn-historikk" onclick="showTab('historikk','bn-historikk')">
    <span class="icon">📈</span>Historikk
  </button>
  <button class="bnav-item" id="bn-makro" onclick="showTab('makro','bn-makro')">
    <span class="icon">🌍</span>Makro
  </button>
</nav>

<!-- ═══════════════════════════ CHAT WIDGET ════════════════════════ -->
<div id="chat-fab" onclick="toggleChat()" title="Spør AI-rådgiveren"
  style="position:fixed;bottom:80px;right:18px;width:54px;height:54px;
  background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:1.4rem;cursor:pointer;box-shadow:0 4px 16px rgba(37,99,235,.45);
  z-index:800;transition:transform .2s" onmouseover="this.style.transform='scale(1.1)'"
  onmouseout="this.style.transform='scale(1)'">💬</div>

<div id="chat-panel" style="display:none;position:fixed;bottom:80px;right:14px;
  width:min(380px,calc(100vw - 28px));height:520px;background:#fff;border-radius:16px;
  box-shadow:0 8px 40px rgba(0,0,0,.22);z-index:801;
  flex-direction:column;overflow:hidden">
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;
    padding:14px 16px;display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-weight:700;font-size:.95rem">🤖 AI Finansrådgiver</div>
      <div style="font-size:.72rem;opacity:.8">Spør om signaler, indikatorer og anbefalinger</div>
    </div>
    <button onclick="toggleChat()" style="background:rgba(255,255,255,.15);border:none;
      color:#fff;border-radius:8px;padding:6px 10px;cursor:pointer;font-size:.85rem">✕</button>
  </div>
  <div id="chat-setup" style="padding:14px">
    <p style="font-size:.8rem;color:#6b7280;margin-bottom:10px">
      Skriv inn din Claude API-nøkkel for å aktivere AI-chat. Nøkkelen lagres lokalt i nettleseren.
    </p>
    <input id="chat-apikey" type="password" placeholder="sk-ant-..."
      style="width:100%;padding:9px 12px;border:1.5px solid #e5e7eb;border-radius:8px;
      font-size:.85rem;margin-bottom:8px"
      onfocus="this.style.borderColor='#2563eb'" onblur="this.style.borderColor='#e5e7eb'">
    <button onclick="saveApiKey()" style="width:100%;padding:9px;background:#2563eb;color:#fff;
      border:none;border-radius:8px;cursor:pointer;font-weight:600;font-size:.85rem">
      Lagre nøkkel og start chat
    </button>
  </div>
  <div id="chat-messages" style="flex:1;overflow-y:auto;padding:12px;display:flex;
    flex-direction:column;gap:10px"></div>
  <div id="chat-input-area" style="padding:10px;border-top:1px solid #e5e7eb;display:flex;gap:8px">
    <input id="chat-input" type="text" placeholder="Spør om en aksje, signal eller strategi..."
      style="flex:1;padding:9px 12px;border:1.5px solid #e5e7eb;border-radius:8px;font-size:.85rem;outline:none"
      onfocus="this.style.borderColor='#2563eb'" onblur="this.style.borderColor='#e5e7eb'"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendChat();}}">
    <button onclick="sendChat()" id="chat-send-btn"
      style="padding:9px 14px;background:#2563eb;color:#fff;border:none;
      border-radius:8px;cursor:pointer;font-size:.9rem;font-weight:600;
      white-space:nowrap;min-width:52px">Send</button>
  </div>
</div>

<!-- ═══════════════════════════ OVERSIKT ═══════════════════════════ -->
<div id="page-oversikt" class="page active">
  <div class="row" id="summary-cards"></div>
  <div id="alert-box"></div>
  <div class="row">
    <div class="card" style="flex:2;min-width:320px">
      <div class="section-title">🎯 Dagens signaler</div>
      <div class="table-wrap"><table><thead><tr>
        <th>Aksje</th><th>Kurs</th><th>Signal</th><th>Score</th><th>Styrke</th>
      </tr></thead>
      <tbody id="signal-overview"></tbody></table></div>
    </div>
    <div class="card" style="flex:1;min-width:220px">
      <div class="section-title">💼 Portefølje</div>
      <div id="port-mini"></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════ SIGNALER ═══════════════════════════ -->
<div id="page-signaler" class="page">
  <div class="card" style="margin-bottom:16px">
    <div class="section-title">🎯 Alle signaler – fullstendig analyse</div>
    <div class="table-wrap"><table><thead><tr>
      <th>Aksje</th><th>Sektor</th><th>Kurs (NOK)</th><th>Endring%</th>
      <th>RSI</th><th>Signal</th><th>Score</th><th>Forsl. NOK</th>
    </tr></thead>
    <tbody id="signals-full"></tbody></table></div>
  </div>
  <div class="card">
    <div class="section-title">💡 Begrunnelser</div>
    <div id="reasons-panel"></div>
  </div>
</div>

<!-- ═══════════════════════════ ANALYSE ═══════════════════════════ -->
<div id="page-analyse" class="page">
  <div style="margin-bottom:16px;display:flex;gap:12px;align-items:center">
    <select id="stock-select" onchange="renderAnalysis()"></select>
    <span style="font-size:.8rem;color:var(--muted)">Velg aksje for detaljert analyse</span>
  </div>
  <div class="chart-wrap">
    <div class="section-title" id="chart-title">Kursgraf</div>
    <canvas id="priceChart" height="100"></canvas>
  </div>
  <div class="row">
    <div class="chart-wrap" style="flex:1;min-width:280px">
      <div class="section-title">RSI (14)</div>
      <canvas id="rsiChart" height="120"></canvas>
    </div>
    <div class="chart-wrap" style="flex:1;min-width:280px">
      <div class="section-title">MACD</div>
      <canvas id="macdChart" height="120"></canvas>
    </div>
  </div>
  <div class="card">
    <div class="section-title" id="ind-title">Nøkkelindikatorer</div>
    <div class="indicator-grid" id="ind-grid"></div>
  </div>
</div>

<!-- ═══════════════════════════ PORTEFØLJE ══════════════════════════ -->
<div id="page-portefolje" class="page">
  <div class="row" id="port-cards"></div>
  <div class="card" style="margin-bottom:16px">
    <div class="section-title">📋 Beholdninger</div>
    <div id="holdings-table"></div>
  </div>
  <div class="card" style="margin-bottom:16px">
    <div class="section-title">➕ Legg til posisjon</div>
    <div class="form-grid" style="margin-bottom:12px">
      <div><label>Aksje</label>
        <select id="buy-symbol" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px"></select>
      </div>
      <div><label>Antall aksjer</label><input type="number" id="buy-shares" placeholder="f.eks. 100"></div>
      <div><label>Kjøpskurs (NOK)</label><input type="number" id="buy-price" placeholder="f.eks. 285.50"></div>
      <div><label>Dato</label><input type="date" id="buy-date"></div>
    </div>
    <button class="primary" onclick="addHolding()">Legg til kjøp</button>
    <span style="font-size:.8rem;color:var(--muted);margin-left:12px" id="buy-msg"></span>
  </div>
  <div class="card">
    <div class="section-title">📜 Transaksjonslogg</div>
    <div id="tx-log"></div>
  </div>
  <div class="card" style="margin-top:16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div>
      <div class="section-title" style="margin-bottom:4px">⚙️ Kapitalinnstillinger</div>
      <div style="font-size:.8rem;color:var(--muted)">Juster tilgjengelig kapital hvis du tilfører eller tar ut penger</div>
    </div>
    <button class="primary" onclick="openCapitalModal()">Juster kapital</button>
  </div>
</div>

<!-- ════════════════════════════ MAKRO ══════════════════════════════ -->
<div id="page-makro" class="page">
  <div class="section-title" style="margin-bottom:12px">🌍 Makroindikatorer</div>
  <div class="macro-grid" id="macro-grid"></div>
  <div class="card" style="margin-top:16px">
    <div class="section-title">ℹ️ Makrotolkning for swing trading</div>
    <div id="macro-comment" style="font-size:.85rem;line-height:1.6;color:var(--muted)"></div>
  </div>
</div>

<!-- ════════════════════════════ HISTORIKK ══════════════════════════ -->
<div id="page-historikk" class="page">
  <div class="row" id="hist-summary-cards"></div>
  <div class="card" style="margin-bottom:16px">
    <div class="section-title">📈 Porteføljeverdi over tid</div>
    <canvas id="histValueChart" height="90"></canvas>
  </div>
  <div class="row">
    <div class="card" style="flex:1;min-width:260px">
      <div class="section-title">💰 Avkastning over tid</div>
      <canvas id="histReturnChart" height="120"></canvas>
    </div>
    <div class="card" style="flex:1;min-width:260px">
      <div class="section-title">🏦 Investert vs. kontanter</div>
      <canvas id="histSplitChart" height="120"></canvas>
    </div>
  </div>
  <div class="card">
    <div class="section-title">📋 Historikklogg</div>
    <div id="hist-table" style="overflow-x:auto"></div>
  </div>
</div>

<script>
// ─── Auth config (SHA-256 hash of "user:pass") ────────────────────────────
const AUTH_HASH = "{auth_hash}";
const AUTH_USER = "{auth_user}";
const SESSION_KEY = "aifin_session";
const SESSION_TTL = 7 * 24 * 60 * 60 * 1000; // 7 days

async function sha256hex(str) {{
  const buf  = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map(b=>b.toString(16).padStart(2,"0")).join("");
}}

async function doLogin() {{
  const user = document.getElementById("login-user").value.trim();
  const pass = document.getElementById("login-pass").value;
  const hash = await sha256hex(user + ":" + pass);
  if (hash === AUTH_HASH) {{
    localStorage.setItem(SESSION_KEY, JSON.stringify({{ts: Date.now()}}));
    showApp();
  }} else {{
    const err = document.getElementById("login-err");
    err.style.display = "block";
    document.getElementById("login-pass").value = "";
    setTimeout(()=>err.style.display="none", 3000);
  }}
}}

function checkSession() {{
  const raw = localStorage.getItem(SESSION_KEY);
  if (!raw) return false;
  try {{
    const {{ts}} = JSON.parse(raw);
    return (Date.now() - ts) < SESSION_TTL;
  }} catch(e) {{ return false; }}
}}

function showApp() {{
  document.getElementById("login-screen").style.display = "none";
  document.getElementById("app-root").style.display     = "block";
}}

// Check existing session on load
if (checkSession()) {{
  showApp();
}} else {{
  document.getElementById("login-screen").style.display = "flex";
}}

// ─── Data ────────────────────────────────────────────────────────────────
const RESULTS   = {results_js};
const PORTFOLIO = {portfolio_js};
const HISTORY   = {history_js};

// Persist portfolio changes in localStorage
const PORT_KEY = "aifin_portfolio";
let portfolio = JSON.parse(localStorage.getItem(PORT_KEY) || "null") || PORTFOLIO;

function savePortfolio() {{
  localStorage.setItem(PORT_KEY, JSON.stringify(portfolio));
}}

// ─── Utilities ───────────────────────────────────────────────────────────
function fmtNOK(v) {{
  if (v == null) return "–";
  return new Intl.NumberFormat("nb-NO",{{style:"currency",currency:"NOK",maximumFractionDigits:0}}).format(v);
}}
function fmtNum(v, dec=2) {{
  if (v == null) return "–";
  return new Intl.NumberFormat("nb-NO",{{minimumFractionDigits:dec,maximumFractionDigits:dec}}).format(v);
}}
function pctSpan(v) {{
  if (v == null) return "–";
  const cls = v > 0 ? "pos" : v < 0 ? "neg" : "neu";
  return `<span class="${{cls}}">${{v > 0 ? "+" : ""}}${{fmtNum(v)}}%</span>`;
}}
function signalBadge(s) {{
  const map = {{"STERKT KJØP":"SK","KJØP":"K","HOLD":"H","SELG":"S","STERKT SELG":"SS","UTILSTREKKELIG DATA":"UD"}};
  const code = map[s] || "H";
  return `<span class="badge badge-${{code}}">${{s}}</span>`;
}}
function scoreBar(score) {{
  const pct = Math.round(((score + 85) / 170) * 100);
  const col  = score >= 40 ? "#16a34a" : score >= 20 ? "#2563eb"
             : score <= -40 ? "#dc2626" : score <= -20 ? "#ea580c" : "#9ca3af";
  return `<div class="signal-bar" style="width:100%;background:#e5e7eb">
    <div style="width:${{pct}}%;height:100%;background:${{col}};border-radius:3px"></div></div>`;
}}

// ─── Tabs (desktop + mobile bottom nav) ──────────────────────────────────
function showTab(id, bnId) {{
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".bnav-item").forEach(b => b.classList.remove("active"));
  document.getElementById("page-"+id).classList.add("active");
  // Desktop tab: mark active via event target
  if (event && event.target && event.target.classList.contains("tab")) {{
    event.target.classList.add("active");
  }}
  // Mobile bottom nav: mark by explicit id
  if (bnId) {{
    const el = document.getElementById(bnId);
    if (el) el.classList.add("active");
  }}
  window.scrollTo(0, 0);
  if (id === "analyse")   renderAnalysis();
  if (id === "historikk") renderHistory();
}}

// ─── Overview cards ──────────────────────────────────────────────────────
function renderSummaryCards() {{
  const p   = portfolio;
  const cap = p.capital || {{}};
  const perf = p.performance || {{}};
  const ret  = perf.total_return_nok || 0;
  const retPct = perf.total_return_pct || 0;
  const retCls = ret >= 0 ? "pos" : "neg";

  const stocks = RESULTS.stocks || [];
  const buys   = stocks.filter(s => ["STERKT KJØP","KJØP"].includes(s.signal)).length;
  const sells  = stocks.filter(s => ["STERKT SELG","SELG"].includes(s.signal)).length;

  document.getElementById("summary-cards").innerHTML = `
    <div class="card">
      <h3>Total Verdi</h3>
      <div class="value">${{fmtNOK(cap.total_value)}}</div>
      <div class="change ${{retCls}}">${{ret>=0?"+":""}}${{fmtNOK(ret)}} (${{retPct>=0?"+":""}}${{fmtNum(retPct)}}%)</div>
    </div>
    <div class="card">
      <h3>Tilgjengelig Kontanter</h3>
      <div class="value">${{fmtNOK(cap.cash)}}</div>
      <div class="change neu">${{fmtNum((cap.cash/(cap.total_value||1))*100)}}% av portefølje</div>
    </div>
    <div class="card">
      <h3>Investert</h3>
      <div class="value">${{fmtNOK(cap.invested)}}</div>
      <div class="change neu">${{(portfolio.holdings||[]).length}} posisjoner</div>
    </div>
    <div class="card">
      <h3>Kjøpssignaler i dag</h3>
      <div class="value pos">${{buys}}</div>
      <div class="change neg">${{sells}} salgssignaler</div>
    </div>
  `;
}}

// ─── Alerts ──────────────────────────────────────────────────────────────
function renderAlerts() {{
  const alerts = RESULTS.alerts || [];
  const box    = document.getElementById("alert-box");
  if (!alerts.length) {{ box.innerHTML = ""; return; }}
  box.innerHTML = alerts.map(a =>
    `<div class="alert ${{a.startsWith("⛔")?"alert-danger":""}}">${{a}}</div>`
  ).join("");
}}

// ─── Signal overview ─────────────────────────────────────────────────────
function renderSignalOverview() {{
  const stocks = RESULTS.stocks || [];
  const sorted = [...stocks].sort((a,b) => b.score - a.score);
  document.getElementById("signal-overview").innerHTML = sorted.map(s => `
    <tr>
      <td><strong>${{s.name}}</strong><br><span style="font-size:.72rem;color:var(--muted)">${{s.symbol}}</span></td>
      <td>${{fmtNum(s.price, 2)}} NOK</td>
      <td>${{signalBadge(s.signal)}}</td>
      <td><strong>${{s.score > 0 ? "+" : ""}}${{s.score}}</strong></td>
      <td style="min-width:80px">${{scoreBar(s.score)}}</td>
    </tr>
  `).join("");
}}

// ─── Portfolio mini ───────────────────────────────────────────────────────
function renderPortMini() {{
  const h = portfolio.holdings || [];
  if (!h.length) {{
    document.getElementById("port-mini").innerHTML =
      `<div class="empty" style="padding:20px">Ingen posisjoner ennå</div>`;
    return;
  }}
  document.getElementById("port-mini").innerHTML = h.map(pos => {{
    const pnl    = pos.unrealized_pnl || 0;
    const pnlPct = pos.unrealized_pnl_pct || 0;
    const cls    = pnl >= 0 ? "pos" : "neg";
    return `<div style="padding:8px 0;border-bottom:1px solid var(--border)">
      <div style="display:flex;justify-content:space-between">
        <strong>${{pos.symbol.replace(".OL","")}}</strong>
        <span class="${{cls}}">${{pnl>=0?"+":""}}${{fmtNum(pnlPct)}}%</span>
      </div>
      <div style="font-size:.75rem;color:var(--muted);margin-top:2px">
        ${{pos.shares}} aksjer · ${{fmtNOK(pos.current_value||pos.cost_basis)}}
      </div>
    </div>`;
  }}).join("");
}}

// ─── Full signals table ───────────────────────────────────────────────────
function renderSignalsFull() {{
  const stocks = RESULTS.stocks || [];
  const sorted = [...stocks].sort((a,b) => b.score - a.score);
  document.getElementById("signals-full").innerHTML = sorted.map(s => {{
    const chg = s.price && s.prev_close
      ? ((s.price/s.prev_close-1)*100).toFixed(2)
      : null;
    const rsiTip = s.rsi != null
      ? (s.rsi < 30 ? "Kraftig oversolgt – potensiell kjøpsmulighet" : s.rsi > 70 ? "Overkjøpt – vurder å ta profitt" : "Nøytral sone")
      : "";
    return `<tr>
      <td><strong>${{s.name}}</strong><br><span style="font-size:.72rem;color:var(--muted)">${{s.symbol}}</span></td>
      <td>${{s.sector||"–"}}</td>
      <td><span class="tip" data-tip="Siste omsatte kurs på Oslo Børs (NOK)">${{fmtNum(s.price, 2)}}</span></td>
      <td>${{chg != null ? pctSpan(parseFloat(chg)) : "–"}}</td>
      <td><span class="tip" data-tip="RSI 0-100: Under 30 = oversolgt (kjøp), over 70 = overkjøpt (selg). ${{rsiTip}}">${{s.rsi != null ? fmtNum(s.rsi,1) : "–"}}</span></td>
      <td><span class="tip" data-tip="Signal basert på kombinasjon av RSI, MACD, Bollinger Bands, trendlinje og volum">${{signalBadge(s.signal)}}</span></td>
      <td><span class="tip" data-tip="Totalscore fra -85 til +85. Over 40 = sterkt kjøp, under -40 = sterkt selg"><strong>${{s.score>0?"+":""}}${{s.score}}</strong></span></td>
      <td><span class="tip" data-tip="Foreslått kjøpsbeløp basert på signal-styrke og tilgjengelig kapital${{s.suggested_nok?" ("+s.suggested_pct+"% av total)":""}}">${{s.suggested_nok ? fmtNOK(s.suggested_nok) : "–"}}</span></td>
    </tr>`;
  }}).join("");

  // Reasons with rationale
  document.getElementById("reasons-panel").innerHTML = sorted
    .filter(s => s.signal !== "UTILSTREKKELIG DATA")
    .map(s => `
      <div style="margin-bottom:16px;padding:12px 14px;background:#f8fafc;border-radius:10px;
        border-left:3px solid ${{s.signal.includes("KJØP")?"#16a34a":s.signal.includes("SELG")?"#dc2626":"#9ca3af"}}">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px">
          <strong>${{s.name}}</strong>
          ${{signalBadge(s.signal)}}
          ${{s.suggested_nok ? `<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:700">${{fmtNOK(s.suggested_nok)}}</span>` : ""}}
        </div>
        ${{s.rationale ? `<p style="font-size:.82rem;color:#374151;margin-bottom:6px">${{s.rationale}}</p>` : ""}}
        ${{(s.reasons&&s.reasons.length) ? `<ul class="reasons-list">${{s.reasons.map(r=>`<li>${{r}}</li>`).join("")}}</ul>` : ""}}
      </div>
    `).join("") || `<div class="empty">Ingen signaler tilgjengelig</div>`;
}}

// ─── Analysis charts ──────────────────────────────────────────────────────
let priceChart, rsiChart, macdChart;

function renderAnalysis() {{
  const sel    = document.getElementById("stock-select");
  const symbol = sel.value;
  const stocks = RESULTS.stocks || [];
  const stock  = stocks.find(s => s.symbol === symbol);
  if (!stock || !stock.history) {{ return; }}

  document.getElementById("chart-title").textContent = `${{stock.name}} (${{symbol}}) – Kursutvikling`;
  document.getElementById("ind-title").textContent    = `${{stock.name}} – Nøkkelindikatorer`;

  const hist   = stock.history;
  const labels = hist.map(h => h.date);
  const closes = hist.map(h => h.close);
  const highs  = hist.map(h => h.high);
  const lows   = hist.map(h => h.low);
  const vols   = hist.map(h => h.volume);

  // Compute indicators
  const sma20  = rollingMean(closes, 20);
  const sma50  = rollingMean(closes, 50);
  const rsiArr = computeRSI(closes, 14);
  const {{macdLine, signalLine, histogram}} = computeMACD(closes);
  const bb     = computeBollinger(closes, 20, 2);

  // ── Price chart ──────────────────────────────────────────────
  if (priceChart) priceChart.destroy();
  priceChart = new Chart(document.getElementById("priceChart"), {{
    type: "line",
    data: {{
      labels,
      datasets: [
        {{ label:"Kurs", data:closes, borderColor:"#2563eb", borderWidth:2,
           pointRadius:0, fill:false, tension:.3 }},
        {{ label:"SMA20", data:sma20, borderColor:"#f59e0b", borderWidth:1.5,
           pointRadius:0, fill:false, borderDash:[4,3], tension:.3 }},
        {{ label:"SMA50", data:sma50, borderColor:"#8b5cf6", borderWidth:1.5,
           pointRadius:0, fill:false, borderDash:[4,3], tension:.3 }},
        {{ label:"BB Øvre", data:bb.upper, borderColor:"rgba(220,38,38,.35)",
           borderWidth:1, pointRadius:0, fill:false, tension:.3 }},
        {{ label:"BB Nedre", data:bb.lower, borderColor:"rgba(22,163,74,.35)",
           borderWidth:1, pointRadius:0, fill:false, tension:.3 }},
      ]
    }},
    options:{{ responsive:true, interaction:{{mode:"index",intersect:false}},
      plugins:{{ legend:{{position:"top",labels:{{boxWidth:12,font:{{size:11}}}}}} }},
      scales:{{ y:{{ grid:{{color:"#f1f5f9"}} }}, x:{{ ticks:{{maxTicksLimit:10}} }} }} }}
  }});

  // ── RSI chart ────────────────────────────────────────────────
  if (rsiChart) rsiChart.destroy();
  rsiChart = new Chart(document.getElementById("rsiChart"), {{
    type:"line",
    data:{{ labels, datasets:[
      {{ label:"RSI", data:rsiArr, borderColor:"#2563eb", borderWidth:2, pointRadius:0, fill:false }}
    ]}},
    options:{{ responsive:true, plugins:{{ legend:{{display:false}},
      annotation:{{}} }},
      scales:{{ y:{{ min:0, max:100,
        grid:{{color:(c)=>c.tick.value===30||c.tick.value===70?"#ef444460":"#f1f5f9"}} }},
        x:{{ ticks:{{maxTicksLimit:10}} }} }} }}
  }});

  // ── MACD chart ───────────────────────────────────────────────
  if (macdChart) macdChart.destroy();
  macdChart = new Chart(document.getElementById("macdChart"), {{
    data:{{ labels, datasets:[
      {{ type:"line",  label:"MACD",   data:macdLine,  borderColor:"#2563eb",
         borderWidth:2, pointRadius:0, fill:false }},
      {{ type:"line",  label:"Signal", data:signalLine, borderColor:"#f59e0b",
         borderWidth:1.5, pointRadius:0, fill:false, borderDash:[4,3] }},
      {{ type:"bar",   label:"Hist",   data:histogram,
         backgroundColor: histogram.map(v=>v>=0?"rgba(22,163,74,.6)":"rgba(220,38,38,.6)") }},
    ]}},
    options:{{ responsive:true,
      plugins:{{ legend:{{position:"top",labels:{{boxWidth:12,font:{{size:11}}}}}} }},
      scales:{{ y:{{ grid:{{color:"#f1f5f9"}} }}, x:{{ ticks:{{maxTicksLimit:10}} }} }} }}
  }});

  // ── Indicator cards ──────────────────────────────────────────
  const ind = stock;
  const rsiNow = rsiArr[rsiArr.length-1];
  const macNow = histogram[histogram.length-1];
  document.getElementById("ind-grid").innerHTML = [
    {{ lbl:"RSI (14)", val: rsiNow!=null?fmtNum(rsiNow,1):"–",
       col: rsiNow<30?"var(--green)":rsiNow>70?"var(--red)":"var(--text)" }},
    {{ lbl:"MACD Hist.", val: macNow!=null?fmtNum(macNow,4):"–",
       col: macNow>0?"var(--green)":macNow<0?"var(--red)":"var(--text)" }},
    {{ lbl:"SMA 20", val: sma20.at(-1)!=null?fmtNum(sma20.at(-1),2):"–", col:"var(--text)" }},
    {{ lbl:"SMA 50", val: sma50.at(-1)!=null?fmtNum(sma50.at(-1),2):"–", col:"var(--text)" }},
    {{ lbl:"ATR%", val: ind.atr_pct!=null?fmtNum(ind.atr_pct,2)+"%":"–", col:"var(--text)" }},
    {{ lbl:"Signal", val: ind.signal, col:"var(--text)" }},
    {{ lbl:"Score", val: (ind.score>0?"+":"")+ind.score, col: ind.score>=20?"var(--green)":ind.score<=-20?"var(--red)":"var(--text)" }},
    {{ lbl:"Forslag", val: ind.suggested_pct?ind.suggested_pct+"%":"Ingen", col:"var(--text)" }},
  ].map(c=>`<div class="ind-card">
    <div class="ind-val" style="color:${{c.col}}">${{c.val}}</div>
    <div class="ind-lbl">${{c.lbl}}</div>
  </div>`).join("");
}}

// ─── Portfolio page ───────────────────────────────────────────────────────
function renderPortfolio() {{
  const cap  = portfolio.capital || {{}};
  const perf = portfolio.performance || {{}};
  const total_return = perf.total_return_nok || 0;
  const retCls = total_return >= 0 ? "pos" : "neg";

  document.getElementById("port-cards").innerHTML = `
    <div class="card">
      <h3>Total Verdi</h3>
      <div class="value">${{fmtNOK(cap.total_value || cap.initial)}}</div>
      <div class="change ${{retCls}}">${{total_return>=0?"+":""}}${{fmtNOK(total_return)}} (${{fmtNum(perf.total_return_pct||0)}}%)</div>
    </div>
    <div class="card">
      <h3>Kontanter</h3>
      <div class="value">${{fmtNOK(cap.cash)}}</div>
      <div class="change neu">${{fmtNum((cap.cash/(cap.total_value||cap.initial||1))*100)}}% av kapital</div>
    </div>
    <div class="card">
      <h3>Urealisert P&L</h3>
      <div class="value ${{(perf.unrealized_pnl||0)>=0?"pos":"neg"}}">${{fmtNOK(perf.unrealized_pnl||0)}}</div>
    </div>
  `;

  const h = portfolio.holdings || [];
  if (!h.length) {{
    document.getElementById("holdings-table").innerHTML =
      `<div class="empty">Ingen åpne posisjoner. Legg til et kjøp nedenfor.</div>`;
  }} else {{
    document.getElementById("holdings-table").innerHTML = `
      <table><thead><tr>
        <th>Aksje</th><th>Antall</th><th>Kjøpskurs</th><th>Nåværende</th>
        <th>Kostnad</th><th>Verdi nå</th><th>P&L</th><th>P&L%</th><th></th>
      </tr></thead><tbody>
      ${{h.map(pos => {{
        const pnl    = pos.unrealized_pnl || 0;
        const pnlPct = pos.unrealized_pnl_pct || 0;
        const cls    = pnl >= 0 ? "pos" : "neg";
        return `<tr>
          <td><strong>${{pos.symbol}}</strong><br><span style="font-size:.72rem;color:var(--muted)">${{pos.name||""}}</span></td>
          <td>${{pos.shares}}</td>
          <td>${{fmtNum(pos.entry_price, 2)}}</td>
          <td>${{fmtNum(pos.current_price||pos.entry_price, 2)}}</td>
          <td>${{fmtNOK(pos.cost_basis)}}</td>
          <td>${{fmtNOK(pos.current_value||pos.cost_basis)}}</td>
          <td class="${{cls}}">${{pnl>=0?"+":""}}${{fmtNOK(pnl)}}</td>
          <td class="${{cls}}">${{pnlPct>=0?"+":""}}${{fmtNum(pnlPct)}}%</td>
          <td><button onclick="sellHolding('${{pos.id}}')" style="background:var(--red);color:#fff;border:none;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:.75rem">Selg</button></td>
        </tr>`;
      }}).join("")}}
      </tbody></table>
    `;
  }}

  // Transaction log
  const tx = portfolio.transactions || [];
  if (!tx.length) {{
    document.getElementById("tx-log").innerHTML = `<div class="empty">Ingen transaksjoner ennå</div>`;
  }} else {{
    const recent = [...tx].reverse().slice(0, 15);
    document.getElementById("tx-log").innerHTML = `
      <table><thead><tr><th>Dato</th><th>Type</th><th>Aksje</th><th>Antall</th><th>Kurs</th><th>Beløp</th></tr></thead>
      <tbody>${{recent.map(t => `<tr>
        <td>${{t.date}}</td>
        <td><span class="badge badge-${{t.type==="KJ"?"K":"S"}}">${{t.type==="KJ"?"KJØP":"SALG"}}</span></td>
        <td>${{t.symbol}}</td><td>${{t.shares}}</td>
        <td>${{fmtNum(t.price,2)}}</td><td>${{fmtNOK(t.amount)}}</td>
      </tr>`).join("")}}</tbody></table>
    `;
  }}
}}

function addHolding() {{
  const sym    = document.getElementById("buy-symbol").value;
  const shares = parseFloat(document.getElementById("buy-shares").value);
  const price  = parseFloat(document.getElementById("buy-price").value);
  const date   = document.getElementById("buy-date").value || new Date().toISOString().slice(0,10);
  const msg    = document.getElementById("buy-msg");

  if (!sym || !shares || !price) {{ msg.textContent = "⚠️ Fyll ut alle felt."; return; }}
  const cost = shares * price;
  if (cost > portfolio.capital.cash) {{ msg.textContent = "⚠️ Ikke nok kontanter."; return; }}

  const stock  = (RESULTS.stocks||[]).find(s=>s.symbol===sym)||{{}};
  const id     = sym+"_"+Date.now();
  portfolio.holdings.push({{
    id, symbol:sym, name:stock.name||sym, shares, entry_price:price,
    current_price:price, cost_basis:round2(cost), current_value:round2(cost),
    unrealized_pnl:0, unrealized_pnl_pct:0, date
  }});
  portfolio.transactions.push({{ type:"KJ", symbol:sym, shares, price, amount:round2(cost), date, id }});
  portfolio.capital.cash = round2(portfolio.capital.cash - cost);
  portfolio.capital.invested = round2(portfolio.capital.invested + cost);
  portfolio.capital.total_value = round2(portfolio.capital.cash + portfolio.capital.invested);
  savePortfolio();
  renderPortfolio();
  renderSummaryCards();
  renderPortMini();
  msg.textContent = `✅ Kjøpt ${{shares}} ${{sym}} à ${{fmtNum(price,2)}} NOK`;
}}

function sellHolding(id) {{
  const idx = portfolio.holdings.findIndex(h=>h.id===id);
  if (idx<0) return;
  const pos  = portfolio.holdings[idx];
  const price = pos.current_price || pos.entry_price;
  const val   = round2(pos.shares * price);
  const pnl   = round2(val - pos.cost_basis);
  portfolio.transactions.push({{
    type:"SG", symbol:pos.symbol, shares:pos.shares, price, amount:val,
    date:new Date().toISOString().slice(0,10), pnl, id
  }});
  portfolio.capital.cash     = round2(portfolio.capital.cash + val);
  portfolio.capital.invested = round2(portfolio.capital.invested - pos.cost_basis);
  portfolio.capital.total_value = round2(portfolio.capital.cash + portfolio.capital.invested);
  portfolio.performance.realized_pnl = round2((portfolio.performance.realized_pnl||0) + pnl);
  portfolio.holdings.splice(idx, 1);
  savePortfolio();
  renderPortfolio();
  renderSummaryCards();
  renderPortMini();
}}

function round2(v) {{ return Math.round(v*100)/100; }}

// ─── Macro page ───────────────────────────────────────────────────────────
function renderMacro() {{
  const macro = RESULTS.macro || [];
  document.getElementById("macro-grid").innerHTML = macro.map(m => {{
    const chg = m.price && m.prev_close
      ? ((m.price/m.prev_close-1)*100).toFixed(2) : null;
    const cls = chg != null ? (parseFloat(chg)>=0?"pos":"neg") : "neu";
    return `<div class="macro-card">
      <h4>${{m.name}}</h4>
      <div class="val">${{m.price!=null?fmtNum(m.price,2):"–"}} ${{m.currency||""}}</div>
      <div class="chg ${{cls}}">${{chg!=null?(parseFloat(chg)>=0?"+":"")+chg+"%":"–"}}</div>
    </div>`;
  }}).join("") || `<div class="empty">Ingen makrodata</div>`;

  // Simple macro comment
  const brent = macro.find(m=>m.symbol==="BZ=F");
  const usdnok = macro.find(m=>m.symbol==="USDNOK=X");
  let comment = "<p>Makromiljøets påvirkning på Oslo Børs:</p><br>";
  if (brent) {{
    const brentChg = brent.price&&brent.prev_close ? ((brent.price/brent.prev_close-1)*100) : 0;
    comment += `<p>🛢️ <strong>Brent råolje (${{fmtNum(brent.price,1)}} USD/fat)</strong>: `;
    if (brentChg > 1) comment += "Stigende oljepris er positivt for Equinor, Aker BP og norsk økonomi generelt.";
    else if (brentChg < -1) comment += "Fallende oljepris gir press på olje- og gassaksjer (EQNR, AKERBP, SUBC).";
    else comment += "Stabil oljepris – ingen sterk retningspåvirkning.";
    comment += "</p><br>";
  }}
  if (usdnok) {{
    const usdChg = usdnok.price&&usdnok.prev_close ? ((usdnok.price/usdnok.prev_close-1)*100) : 0;
    comment += `<p>💱 <strong>USD/NOK (${{fmtNum(usdnok.price,2)}})</strong>: `;
    if (usdChg > 0.3) comment += "Svakere NOK kan styrke eksportbedrifter (Equinor, Mowi, Yara).";
    else if (usdChg < -0.3) comment += "Sterkere NOK kan dempe inntjeningen for eksportrettede selskaper.";
    else comment += "Stabil kurs – nøytral påvirkning.";
    comment += "</p>";
  }}
  document.getElementById("macro-comment").innerHTML = comment;
}}

// ─── Technical indicator helpers (in-browser) ─────────────────────────────
function rollingMean(data, n) {{
  return data.map((_,i) => {{
    if (i < n-1) return null;
    const slice = data.slice(i-n+1, i+1);
    return slice.reduce((a,b)=>a+b,0)/n;
  }});
}}
function computeRSI(data, n=14) {{
  const gains=[],losses=[];
  for(let i=1;i<data.length;i++) {{
    const d=data[i]-data[i-1];
    gains.push(d>0?d:0); losses.push(d<0?-d:0);
  }}
  const rsi=[...Array(n).fill(null)];
  let ag=gains.slice(0,n).reduce((a,b)=>a+b,0)/n;
  let al=losses.slice(0,n).reduce((a,b)=>a+b,0)/n;
  rsi.push(100-100/(1+ag/(al||1e-9)));
  for(let i=n;i<gains.length;i++) {{
    ag=(ag*(n-1)+gains[i])/n; al=(al*(n-1)+losses[i])/n;
    rsi.push(100-100/(1+ag/(al||1e-9)));
  }}
  return rsi;
}}
function computeMACD(data, fast=12, slow=26, sig=9) {{
  const ema = (d,s)=>d.reduce((acc,v,i)=>{{
    if(i===0)return[v];
    const k=2/(s+1); acc.push(v*k+acc[i-1]*(1-k)); return acc;
  }},[]);
  const ef=ema(data,fast), es=ema(data,slow);
  const macdLine=data.map((_,i)=>i<slow-1?null:(ef[i]-es[i]));
  const validMacd=macdLine.filter(v=>v!==null);
  const sigArr=ema(validMacd,sig);
  const fullSig=[...Array(slow-1).fill(null),...sigArr.slice(0,-(sig-1)||undefined),...Array(sig-1).fill(null)];
  // Rebuild properly
  const signalLine=[];
  let j=0;
  for(let i=0;i<macdLine.length;i++) {{
    if(macdLine[i]===null){{signalLine.push(null);continue;}}
    signalLine.push(sigArr[j]??null); j++;
  }}
  const histogram=macdLine.map((v,i)=>v!=null&&signalLine[i]!=null?v-signalLine[i]:null);
  return {{macdLine,signalLine,histogram}};
}}
function computeBollinger(data, n=20, std=2) {{
  const upper=[],lower=[];
  for(let i=0;i<data.length;i++) {{
    if(i<n-1){{upper.push(null);lower.push(null);continue;}}
    const sl=data.slice(i-n+1,i+1);
    const mean=sl.reduce((a,b)=>a+b,0)/n;
    const sd=Math.sqrt(sl.reduce((a,b)=>a+(b-mean)**2,0)/n);
    upper.push(mean+std*sd); lower.push(mean-std*sd);
  }}
  return {{upper,lower}};
}}

// ─── Capital adjustment ───────────────────────────────────────────────────
function openCapitalModal() {{
  document.getElementById("adj-initial").value = portfolio.capital?.initial || "";
  document.getElementById("adj-cash").value    = portfolio.capital?.cash    || "";
  document.getElementById("capital-modal").classList.add("open");
}}
function closeCapitalModal() {{
  document.getElementById("capital-modal").classList.remove("open");
}}
function saveCapital() {{
  const newInitial = parseFloat(document.getElementById("adj-initial").value);
  const newCash    = parseFloat(document.getElementById("adj-cash").value);
  if (isNaN(newInitial) || isNaN(newCash)) {{
    alert("Fyll inn gyldige tall for begge feltene.");
    return;
  }}
  portfolio.capital.initial     = newInitial;
  portfolio.capital.cash        = newCash;
  portfolio.capital.total_value = Math.round((newCash + (portfolio.capital.invested||0))*100)/100;
  portfolio.performance.total_return_nok = Math.round((portfolio.capital.total_value - newInitial)*100)/100;
  portfolio.performance.total_return_pct = Math.round(((portfolio.capital.total_value/newInitial)-1)*10000)/100;
  savePortfolio();
  closeCapitalModal();
  renderSummaryCards();
  renderPortfolio();
  renderPortMini();
  alert("✅ Kapital oppdatert!");
}}
// Close modal on background click
document.getElementById("capital-modal").addEventListener("click", function(e) {{
  if (e.target === this) closeCapitalModal();
}});

// ─── History charts ───────────────────────────────────────────────────────
let histValueChart, histReturnChart, histSplitChart;
function renderHistory() {{
  const snaps = HISTORY || [];
  if (!snaps.length) {{
    ["hist-summary-cards","hist-table"].forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.innerHTML = `<div class="empty">Ingen historikkdata ennå – kjør agent.py daglig for å bygge opp historikk.</div>`;
    }});
    return;
  }}

  const labels    = snaps.map(s => s.date);
  const totals    = snaps.map(s => s.total_value);
  const retPcts   = snaps.map(s => s.return_pct);
  const invested  = snaps.map(s => s.invested);
  const cash      = snaps.map(s => s.cash);
  const initial   = snaps[0]?.initial || snaps[0]?.total_value || 100000;

  // Summary cards
  const first = snaps[0], last = snaps[snaps.length-1];
  const best  = snaps.reduce((a,b) => b.return_pct > a.return_pct ? b : a, first);
  const worst = snaps.reduce((a,b) => b.return_pct < a.return_pct ? b : a, first);
  const retC  = last.return_nok >= 0 ? "pos" : "neg";
  document.getElementById("hist-summary-cards").innerHTML = `
    <div class="card">
      <h3>Total avkastning</h3>
      <div class="value ${{retC}}">${{last.return_pct>=0?"+":""}}${{fmtNum(last.return_pct)}}%</div>
      <div class="change ${{retC}}">${{last.return_nok>=0?"+":""}}${{fmtNOK(last.return_nok)}}</div>
    </div>
    <div class="card">
      <h3>Startkapital</h3>
      <div class="value">${{fmtNOK(initial)}}</div>
      <div class="change neu">Investert ${{snaps.length}} dager</div>
    </div>
    <div class="card">
      <h3>Beste dag</h3>
      <div class="value pos">+${{fmtNum(best.return_pct)}}%</div>
      <div class="change neu">${{best.date}}</div>
    </div>
    <div class="card">
      <h3>Svakeste dag</h3>
      <div class="value neg">${{worst.return_pct>=0?"+":""}}${{fmtNum(worst.return_pct)}}%</div>
      <div class="change neu">${{worst.date}}</div>
    </div>
  `;

  // Portfolio value chart
  if (histValueChart) histValueChart.destroy();
  histValueChart = new Chart(document.getElementById("histValueChart"), {{
    type: "line",
    data: {{
      labels,
      datasets: [
        {{ label:"Porteføljeverdi", data:totals, borderColor:"#2563eb",
           borderWidth:2.5, pointRadius:2, fill:true,
           backgroundColor:"rgba(37,99,235,.08)", tension:.3 }},
        {{ label:"Startkapital", data:snaps.map(()=>initial),
           borderColor:"#9ca3af", borderWidth:1.5, borderDash:[5,4],
           pointRadius:0, fill:false }},
      ]
    }},
    options:{{ responsive:true, interaction:{{mode:"index",intersect:false}},
      plugins:{{ legend:{{position:"top",labels:{{boxWidth:12,font:{{size:11}}}}}} }},
      scales:{{ y:{{ ticks:{{callback:v=>fmtNOK(v)}}, grid:{{color:"#f1f5f9"}} }},
        x:{{ticks:{{maxTicksLimit:8}}}} }} }}
  }});

  // Return % chart
  if (histReturnChart) histReturnChart.destroy();
  histReturnChart = new Chart(document.getElementById("histReturnChart"), {{
    type:"bar",
    data:{{ labels, datasets:[{{
      label:"Avkastning %",
      data:retPcts,
      backgroundColor:retPcts.map(v=>v>=0?"rgba(22,163,74,.7)":"rgba(220,38,38,.7)"),
      borderRadius:3
    }}]}},
    options:{{ responsive:true, plugins:{{legend:{{display:false}}}},
      scales:{{ y:{{ticks:{{callback:v=>v+"%"}},grid:{{color:"#f1f5f9"}}}},
        x:{{ticks:{{maxTicksLimit:8}}}} }} }}
  }});

  // Invested vs cash stacked
  if (histSplitChart) histSplitChart.destroy();
  histSplitChart = new Chart(document.getElementById("histSplitChart"), {{
    type:"bar",
    data:{{ labels, datasets:[
      {{ label:"Investert", data:invested, backgroundColor:"rgba(37,99,235,.75)", borderRadius:0 }},
      {{ label:"Kontanter", data:cash,     backgroundColor:"rgba(22,163,74,.45)",  borderRadius:0 }},
    ]}},
    options:{{ responsive:true, plugins:{{legend:{{position:"top",labels:{{boxWidth:12,font:{{size:11}}}}}}}},
      scales:{{ x:{{stacked:true,ticks:{{maxTicksLimit:8}}}},
        y:{{stacked:true,ticks:{{callback:v=>fmtNOK(v)}},grid:{{color:"#f1f5f9"}}}} }} }}
  }});

  // History table
  const rows = [...snaps].reverse().slice(0, 30);
  document.getElementById("hist-table").innerHTML = `
    <table><thead><tr>
      <th>Dato</th>
      <th><span class="tip" data-tip="Total porteføljeverdi (investert + kontanter)">Total verdi</span></th>
      <th><span class="tip" data-tip="Hvor mye er investert i aksjer">Investert</span></th>
      <th><span class="tip" data-tip="Ledige kontanter">Kontanter</span></th>
      <th><span class="tip" data-tip="Avkastning i NOK fra startkapital">Avk. NOK</span></th>
      <th><span class="tip" data-tip="Avkastning i prosent fra startkapital">Avk. %</span></th>
      <th>Pos.</th>
    </tr></thead><tbody>
    ${{rows.map(s => {{
      const rc = s.return_nok >= 0 ? "pos" : "neg";
      return `<tr>
        <td>${{s.date}}</td>
        <td>${{fmtNOK(s.total_value)}}</td>
        <td>${{fmtNOK(s.invested)}}</td>
        <td>${{fmtNOK(s.cash)}}</td>
        <td class="${{rc}}">${{s.return_nok>=0?"+":""}}${{fmtNOK(s.return_nok)}}</td>
        <td class="${{rc}}">${{s.return_pct>=0?"+":""}}${{fmtNum(s.return_pct)}}%</td>
        <td>${{s.n_positions}}</td>
      </tr>`;
    }}).join("")}}
    </tbody></table>`;
}}

// ─── Init ─────────────────────────────────────────────────────────────────
function init() {{
  const stocks = RESULTS.stocks || [];

  // Populate selects
  const sel1 = document.getElementById("stock-select");
  const sel2 = document.getElementById("buy-symbol");
  stocks.forEach(s => {{
    [sel1,sel2].forEach(sel => {{
      const opt = document.createElement("option");
      opt.value = s.symbol; opt.textContent = `${{s.name}} (${{s.symbol}})`;
      sel.appendChild(opt);
    }});
  }});

  // Set today's date default
  document.getElementById("buy-date").value = new Date().toISOString().slice(0,10);

  // Sync portfolio with latest prices from RESULTS
  const quotes = {{}};
  stocks.forEach(s => {{
    quotes[s.symbol] = {{ price: s.price, prev_close: s.prev_close }};
  }});
  (portfolio.holdings || []).forEach(h => {{
    const q = quotes[h.symbol];
    if (q && q.price) {{
      h.current_price       = q.price;
      h.current_value       = Math.round(h.shares * q.price * 100)/100;
      h.unrealized_pnl      = Math.round((h.current_value - h.cost_basis)*100)/100;
      h.unrealized_pnl_pct  = Math.round((h.current_price/h.entry_price-1)*10000)/100;
    }}
  }});
  const inv = (portfolio.holdings||[]).reduce((a,h)=>a+(h.current_value||h.cost_basis),0);
  portfolio.capital.invested   = Math.round(inv*100)/100;
  portfolio.capital.total_value = Math.round((portfolio.capital.cash+inv)*100)/100;
  savePortfolio();

  renderSummaryCards();
  renderAlerts();
  renderSignalOverview();
  renderPortMini();
  renderSignalsFull();
  renderPortfolio();
  renderMacro();
  renderHistory();
}}

init();

// ─── Chat widget ──────────────────────────────────────────────────────────
const APIKEY_KEY = "aifin_apikey";
let chatOpen = false;
let chatHistory = [];

function toggleChat() {{
  chatOpen = !chatOpen;
  const panel = document.getElementById("chat-panel");
  panel.style.display = chatOpen ? "flex" : "none";
  if (chatOpen) {{
    const key = localStorage.getItem(APIKEY_KEY);
    if (key) {{
      document.getElementById("chat-setup").style.display = "none";
      document.getElementById("chat-messages").style.display = "flex";
      document.getElementById("chat-input-area").style.display = "flex";
      if (!chatHistory.length) addBotMessage("Hei! Jeg er din AI finansrådgiver. Du kan spørre meg om signaler, indikatorer, enkeltaksjer eller strategi. Hva lurer du på?");
    }} else {{
      document.getElementById("chat-setup").style.display = "block";
      document.getElementById("chat-messages").style.display = "none";
      document.getElementById("chat-input-area").style.display = "none";
    }}
    setTimeout(()=>document.getElementById("chat-input").focus(), 100);
  }}
}}

function saveApiKey() {{
  const key = document.getElementById("chat-apikey").value.trim();
  if (!key.startsWith("sk-")) {{
    alert("API-nøkkel må starte med 'sk-'"); return;
  }}
  localStorage.setItem(APIKEY_KEY, key);
  document.getElementById("chat-setup").style.display = "none";
  document.getElementById("chat-messages").style.display = "flex";
  document.getElementById("chat-input-area").style.display = "flex";
  addBotMessage("Nøkkel lagret! Hei! Jeg er din AI finansrådgiver. Spør meg om signaler, aksjer eller strategi.");
}}

function addBotMessage(text) {{
  chatHistory.push({{role:"assistant", content:text}});
  renderMessages();
}}

function renderMessages() {{
  const box = document.getElementById("chat-messages");
  box.innerHTML = chatHistory.map(m => {{
    const isUser = m.role === "user";
    return `<div style="display:flex;justify-content:${{isUser?"flex-end":"flex-start"}}">
      <div style="max-width:88%;padding:10px 13px;border-radius:${{isUser?"14px 14px 4px 14px":"14px 14px 14px 4px"}};
        background:${{isUser?"#2563eb":"#f1f5f9"}};color:${{isUser?"#fff":"#1a2035"}};
        font-size:.85rem;line-height:1.5;white-space:pre-wrap">${{escHtml(m.content)}}</div>
    </div>`;
  }}).join("");
  box.scrollTop = box.scrollHeight;
}}

function escHtml(s) {{
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

async function sendChat() {{
  const input = document.getElementById("chat-input");
  const msg   = input.value.trim();
  if (!msg) return;
  const key = localStorage.getItem(APIKEY_KEY);
  if (!key) {{ toggleChat(); return; }}

  chatHistory.push({{role:"user", content:msg}});
  renderMessages();
  input.value = "";
  document.getElementById("chat-send-btn").textContent = "⏳";
  document.getElementById("chat-send-btn").disabled = true;

  // Build context summary
  const stocks   = RESULTS.stocks || [];
  const buys     = stocks.filter(s=>["STERKT KJØP","KJØP"].includes(s.signal));
  const sells    = stocks.filter(s=>["STERKT SELG","SELG"].includes(s.signal));
  const port     = portfolio;
  const context  = `Du er en norsk AI finansrådgiver for Oslo Børs swing trading.

DAGENS ANALYSE (generert ${{RESULTS.generated_at||"ukjent"}}):
Kjøpssignaler (${{buys.length}}): ${{buys.map(s=>`${{s.name}} score=${{s.score>0?"+":""}}${{s.score}}${{s.suggested_nok?" forslag="+(s.suggested_nok).toLocaleString("nb-NO")+"NOK":""}}`).join(", ")||"ingen"}}
Salgssignaler (${{sells.length}}): ${{sells.map(s=>`${{s.name}} score=${{s.score}}`).join(", ")||"ingen"}}

PORTEFØLJE: Totalt ${{fmtNOK(port.capital?.total_value)}}, Kontanter ${{fmtNOK(port.capital?.cash)}}, ${{(port.holdings||[]).length}} posisjoner

FULL AKSJELISTE (topp 20 etter score):
${{[...stocks].sort((a,b)=>b.score-a.score).slice(0,20).map(s=>`- ${{s.name}} (${{s.symbol}}): ${{s.signal}}, score=${{s.score>0?"+":""}}${{s.score}}, RSI=${{s.rsi!=null?s.rsi.toFixed(1):"–"}}, kurs=${{s.price}} NOK`).join("\\n")}}

Svar på norsk, kort og konkret. Unngå ansvarsfraskrivelser unntatt hvis nødvendig.`;

  try {{
    const resp = await fetch("https://api.anthropic.com/v1/messages", {{
      method: "POST",
      headers: {{
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
        "content-type": "application/json"
      }},
      body: JSON.stringify({{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 600,
        system: context,
        messages: chatHistory.filter(m=>m.role!=="system")
      }})
    }});
    if (!resp.ok) {{
      const err = await resp.json().catch(()=>({{}}));
      throw new Error(err.error?.message || `HTTP ${{resp.status}}`);
    }}
    const data = await resp.json();
    addBotMessage(data.content[0].text);
  }} catch(e) {{
    addBotMessage(`⚠️ Feil: ${{e.message}}\\n\\nSjekk at API-nøkkelen er gyldig (gå til console.anthropic.com).`);
  }}

  document.getElementById("chat-send-btn").textContent = "Send";
  document.getElementById("chat-send-btn").disabled = false;
  document.getElementById("chat-input").focus();
}}

</script>
</div><!-- end app-root -->
</body>
</html>"""
    return html


# ─── Main ─────────────────────────────────────────────────────────────────────

def is_oslo_bors_open() -> bool:
    """Return True if Oslo Børs is open for trading today.

    Uses pandas_market_calendars when available; otherwise falls back to a
    hardcoded list of Norwegian public holidays so the check still works even
    without the library installed.
    """
    today = datetime.date.today()

    # Always skip weekends (belt-and-suspenders, calendar does this too)
    if today.weekday() >= 5:
        return False

    try:
        import pandas_market_calendars as mcal  # type: ignore
        ose = mcal.get_calendar("OSE")
        schedule = ose.schedule(
            start_date=today.isoformat(),
            end_date=today.isoformat(),
        )
        return not schedule.empty

    except ImportError:
        pass  # library not installed – use hardcoded fallback

    # ── Hardcoded Norwegian public holidays (fixed + moveable) ──────────────
    year = today.year

    # Fixed-date holidays
    fixed = {
        (1,  1),   # Nyttårsdag
        (5,  1),   # Arbeidernes dag
        (5, 17),   # Grunnlovsdagen
        (12, 24),  # Julaften (Oslo Børs closes at 13:00; treat as closed)
        (12, 25),  # Første juledag
        (12, 26),  # Andre juledag
        (12, 31),  # Nyttårsaften (closes at 13:00; treat as closed)
    }
    if (today.month, today.day) in fixed:
        return False

    # Moveable Easter-based holidays (Easter Sunday by Anonymous Gregorian)
    def easter_sunday(y: int) -> datetime.date:
        a = y % 19
        b, c = divmod(y, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month, day = divmod(114 + h + l - 7 * m, 31)
        return datetime.date(y, month, day + 1)

    e = easter_sunday(year)
    moveable_offsets = [-3, -2, 0, 1, 39, 49, 50]  # Skjærtorsdag…2. Pinsedag
    moveable = {e + datetime.timedelta(days=d) for d in moveable_offsets}
    if today in moveable:
        return False

    return True


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   AI Finansrådgiver – Oslo Børs Agent v1.0   ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── Market-open guard ──────────────────────────────────────────────────
    if not is_oslo_bors_open():
        today_str = datetime.date.today().strftime("%A %d.%m.%Y")
        print(f"⏸  Oslo Børs er stengt i dag ({today_str}). Avslutter.")
        print("   (Børsen er stengt i helger og på norske helligdager.)\n")
        sys.exit(0)

    watchlist = load_json(WATCHLIST_FILE)
    portfolio = load_json(PORTFOLIO_FILE)
    cfg       = watchlist.get("settings", {})
    auth_cfg  = cfg.get("auth", {"username": "eyvind", "password": "oslo2026"})

    stocks_config = watchlist.get("stocks", [])
    macro_config  = watchlist.get("macro",  [])
    lookback      = cfg.get("lookback_days", 90)

    results = {
        "generated_at": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
        "stocks": [],
        "macro":  [],
        "alerts": [],
    }

    # ── Stock analysis ─────────────────────────────────────────────────────
    print(f"📊 Henter og analyserer {len(stocks_config)} aksjer...\n")
    quotes = {}
    for sc in stocks_config:
        sym  = sc["symbol"]
        name = sc["name"]
        print(f"   {sym:15s} {name}")

        df = fetch_stock(sym, lookback)
        if df is None or len(df) < 20:
            continue

        q    = fetch_quote(sym)
        fund = fetch_fundamentals(sym)
        sig  = generate_signal(df, cfg)
        hist = price_history_to_list(df, 60)

        price      = q.get("price") or (df["Close"].iloc[-1])
        prev_close = q.get("prev_close") or (df["Close"].iloc[-2] if len(df) > 1 else price)

        rsi_val = sig["indicators"].get("rsi")
        entry = {
            "symbol":        sym,
            "name":          name,
            "sector":        sc.get("sector", fund.get("sector", "")),
            "price":         round(float(price), 2) if price else None,
            "prev_close":    round(float(prev_close), 2) if prev_close else None,
            "signal":        sig["signal"],
            "score":         sig["score"],
            "suggested_pct": sig["suggested_pct"],
            "suggested_nok": calculate_suggested_nok(sig["signal"], sig["score"], portfolio),
            "rationale":     generate_rationale(sig["signal"], sig["score"], sig["reasons"], rsi_val),
            "reasons":       sig["reasons"],
            "rsi":           rsi_val,
            "atr_pct":       sig["indicators"].get("atr_pct"),
            "pe_ratio":      fund.get("pe_ratio"),
            "div_yield_pct": fund.get("div_yield_pct"),
            "history":       hist,
        }
        results["stocks"].append(entry)
        quotes[sym] = {"price": price, "prev_close": prev_close}

        badge = {"STERKT KJØP": "🟢🟢", "KJØP": "🟢", "HOLD": "⚪",
                 "SELG": "🔴", "STERKT SELG": "🔴🔴"}.get(sig["signal"], "❓")
        print(f"      {badge} {sig['signal']:15s} score={sig['score']:+.0f}  RSI={sig['indicators'].get('rsi', '–')}")

    # ── Macro data ─────────────────────────────────────────────────────
    print("\n🌍 Henter makrodata...")
    for mc in macro_config:
        sym  = mc["symbol"]
        df   = fetch_stock(sym, 5)
        if df is None or df.empty:
            results["macro"].append({"symbol": sym, "name": mc["name"],
                                     "price": None, "prev_close": None, "currency": ""})
            continue
        price      = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        q          = fetch_quote(sym)
        results["macro"].append({
            "symbol":     sym,
            "name":       mc["name"],
            "type":       mc.get("type"),
            "price":      round(price, 2),
            "prev_close": round(prev_close, 2),
            "currency":   q.get("currency", ""),
        })
        chg = (price / prev_close - 1) * 100 if prev_close else 0
        print(f"   {sym:12s} {price:10.2f}  {chg:+.2f}%")

    # ── Update portfolio ─────────────────────────────────────────────────────
    portfolio = update_portfolio_values(portfolio, quotes)
    alerts    = check_stop_loss_take_profit(portfolio)
    results["alerts"] = alerts
    if alerts:
        print("\n🚨 PORTEFØLJE VARSLER:")
        for a in alerts:
            print(f"   {a}")

    # ── Save results & history ──────────────────────────────────────────────────
    save_json(RESULTS_FILE, results)
    save_json(PORTFOLIO_FILE, portfolio)
    history = update_portfolio_history(portfolio)
    print(f"   📅 Historikk: {len(history.get('snapshots',[]))} daglige snapshots lagret")

    # ── Generate HTML dashboard ───────────────────────────────────────────────
    html = generate_html_dashboard(results, portfolio, history, auth=auth_cfg)
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Ferdig!")
    print(f"   📁 Resultater: {RESULTS_FILE.name}")
    print(f"   📊 Dashboard:  {DASHBOARD_FILE.name}  ← åpne i nettleser")

    # ── Top picks summary ──────────────────────────────────────────────────────
    buys  = [s for s in results["stocks"] if s["signal"] in ("STERKT KJØP", "KJØP")]
    sells = [s for s in results["stocks"] if s["signal"] in ("STERKT SELG", "SELG")]
    buys.sort(key=lambda x: -x["score"])
    if buys:
        print("\n🟢 KJØPSANBEFALINGER:")
        for s in buys:
            print(f"   {s['symbol']:15s} {s['signal']:15s} score={s['score']:+.0f}  forslag={s['suggested_pct']}%")
    if sells:
        print("\n🔴 SALGSANBEFALINGER:")
        for s in sells:
            print(f"   {s['symbol']:15s} {s['signal']:15s} score={s['score']:+.0f}")

    print()


if __name__ == "__main__":
    main()
