"""
╔══════════════════════════════════════════════════════════════╗
║       GOLD SIGNAL BOT v4.0 — Day Trading Edition            ║
║       Render/Railway · cloud 24/7                           ║
╚══════════════════════════════════════════════════════════════╝

Variables d'environnement requises :
  TELEGRAM_TOKEN   — token de ton bot BotFather
  TELEGRAM_CHAT_ID — ton chat ID Telegram
  TWELVE_DATA_KEY  — clé API Twelve Data (twelvedata.com gratuit)

Variables optionnelles :
  SCAN_INTERVAL      — secondes entre chaque scan (défaut: 900 = 15 min)
  COOLDOWN_MINUTES   — délai min entre 2 signaux (défaut: 180 = 3h)
  MIN_SCORE_15M      — confluence requise 15m (défaut: 4)
  MIN_SCORE_1H       — confluence requise 1h  (défaut: 3)
  MIN_SCORE_SELL_15M — seuil SELL 15m         (défaut: 3)
  MIN_SCORE_SELL_1H  — seuil SELL 1h          (défaut: 2)

v4.0 — Day Trading :
  - Timeframes 15m + 1h (au lieu de 1m + 5m)
  - RSI à 40/60 (moins extrême, capte les retournements de tendance)
  - Stochastique à 20/80 (standard day trading)
  - SL = 2×ATR · TP1 = 1.5× · TP2 = 3× · TP3 = 5× (objectifs ambitieux)
  - Zone d'entrée ±0.3×ATR
  - Scan toutes les 15 min → ~50 appels/jour → couverture 07h-21h Paris
  - Cooldown 3h entre signaux (1-3 trades/jour max)
  - Prix temps réel Finnhub pour niveaux précis
"""

import os
import time
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY", "d8k2b71r01qjgd6qig80d8k2b71r01qjgd6qig8g")

SCAN_INTERVAL_SEC   = int(os.environ.get("SCAN_INTERVAL", "900"))    # 15 minutes
COOLDOWN_MINUTES    = int(os.environ.get("COOLDOWN_MINUTES", "180"))  # 3 heures

MIN_SCORE_15M       = int(os.environ.get("MIN_SCORE_15M", "4"))
MIN_SCORE_1H        = int(os.environ.get("MIN_SCORE_1H",  "3"))
MIN_SCORE_SELL_15M  = int(os.environ.get("MIN_SCORE_SELL_15M", "3"))
MIN_SCORE_SELL_1H   = int(os.environ.get("MIN_SCORE_SELL_1H",  "2"))

BARS_15M = 100   # 100 × 15 min = 25h d'historique
BARS_1H  = 72    # 72 × 1h     = 3 jours d'historique

# ═══════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GoldBot")

# ═══════════════════════════════════════════════════
#  INDICATEURS TECHNIQUES
# ═══════════════════════════════════════════════════

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

def stochastic(high, low, close, k=14, d=3):
    lowest_low    = low.rolling(k).min()
    highest_high  = high.rolling(k).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

# ═══════════════════════════════════════════════════
#  PRIX TEMPS RÉEL — Finnhub
# ═══════════════════════════════════════════════════

def get_realtime_price():
    try:
        resp = requests.get(
            f"https://finnhub.io/api/v1/quote?symbol=OANDA:XAU_USD&token={FINNHUB_KEY}",
            timeout=5
        )
        data  = resp.json()
        price = data.get("c")
        if price and float(price) > 1000:
            log.info(f"  Prix temps réel (Finnhub): ${float(price):,.2f}")
            return float(price)
        log.warning(f"  Finnhub quote inattendu: {data}")
        return None
    except Exception as e:
        log.warning(f"  Finnhub erreur: {e} — fallback Twelve Data")
        return None

# ═══════════════════════════════════════════════════
#  ANALYSE & SCORING (optimisé day trading)
# ═══════════════════════════════════════════════════

def compute_signals(df, timeframe="15m"):
    close, high, low = df["Close"], df["High"], df["Low"]

    rsi_val = rsi(close).iloc[-1]
    _, _, histogram = macd(close)
    hist_val  = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2]
    ema9  = ema(close, 9).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    bb_u  = bb_upper.iloc[-1]
    bb_l  = bb_lower.iloc[-1]
    bb_m  = bb_mid.iloc[-1]
    price = close.iloc[-1]
    stoch_k, stoch_d = stochastic(high, low, close)
    sk      = stoch_k.iloc[-1]; sd      = stoch_d.iloc[-1]
    sk_prev = stoch_k.iloc[-2]; sd_prev = stoch_d.iloc[-2]
    atr_val = atr(high, low, close).iloc[-1]

    votes = {}

    # ── RSI — seuils 40/60 pour day trading ──
    # (moins extrême que scalping 35/65 → capte les retournements plus tôt)
    if rsi_val < 40:   votes["RSI"] = ( 1, f"RSI {rsi_val:.1f} — Survendu ✅")
    elif rsi_val > 60: votes["RSI"] = (-1, f"RSI {rsi_val:.1f} — Suracheté ✅")
    else:              votes["RSI"] = ( 0, f"RSI {rsi_val:.1f} — Neutre")

    # ── MACD ──
    if hist_prev < 0 and hist_val >= 0:
        votes["MACD"] = ( 1, f"MACD croisement haussier ✅")
    elif hist_prev > 0 and hist_val <= 0:
        votes["MACD"] = (-1, f"MACD croisement baissier ✅")
    elif hist_val > 0 and hist_val < hist_prev:
        votes["MACD"] = ( 0, f"MACD haussier momentum baissier ({hist_val:.3f})")
    elif hist_val < 0 and hist_val > hist_prev:
        votes["MACD"] = ( 0, f"MACD baissier momentum haussier ({hist_val:.3f})")
    elif hist_val > 0:
        votes["MACD"] = ( 1, f"MACD haussier ({hist_val:.3f})")
    else:
        votes["MACD"] = (-1, f"MACD baissier ({hist_val:.3f})")

    # ── EMA 9/21 — croisement de tendance ──
    if ema9 > ema21:   votes["EMA9/21"] = ( 1, f"EMA9 > EMA21 — Tendance haussière ✅")
    else:              votes["EMA9/21"] = (-1, f"EMA9 < EMA21 — Tendance baissière ✅")

    # ── Bollinger Bands ──
    bb_range = bb_u - bb_l
    bb_pct = (price - bb_l) / bb_range if bb_range > 0 else 0.5
    if bb_pct < 0.2:   votes["BB"] = ( 1, f"Prix bande basse BB — Rebond potentiel ✅")
    elif bb_pct > 0.8: votes["BB"] = (-1, f"Prix bande haute BB — Résistance ✅")
    elif price > bb_m: votes["BB"] = ( 1, f"Prix au-dessus BB mid")
    else:              votes["BB"] = (-1, f"Prix sous BB mid")

    # ── Stochastique — seuils 20/80 pour day trading ──
    if sk < 20 and sk > sd and sk_prev <= sd_prev:
        votes["Stoch"] = ( 1, f"Stoch croisement haussier survendu ✅")
    elif sk > 80 and sk < sd and sk_prev >= sd_prev:
        votes["Stoch"] = (-1, f"Stoch croisement baissier suracheté ✅")
    elif sk < 20: votes["Stoch"] = ( 1, f"Stoch survendu ({sk:.1f}) ✅")
    elif sk > 80: votes["Stoch"] = (-1, f"Stoch suracheté ({sk:.1f}) ✅")
    else:         votes["Stoch"] = ( 0, f"Stoch neutre ({sk:.1f})")

    # ── EMA 50 — filtre de tendance de fond ──
    diff_pct = (price - ema50) / ema50 * 100
    if abs(diff_pct) < 0.15:
        votes["EMA50"] = ( 0, f"Prix ≈ EMA50 ({diff_pct:+.2f}%) — Zone neutre")
    elif price > ema50:
        votes["EMA50"] = ( 1, f"Prix > EMA50 — Tendance haussière ✅")
    else:
        votes["EMA50"] = (-1, f"Prix < EMA50 — Tendance baissière ✅")

    score = sum(v[0] for v in votes.values())

    if score >= MIN_SCORE_15M:
        direction = "BUY"
    elif score <= -MIN_SCORE_SELL_15M:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    return {
        "score":     score,
        "direction": direction,
        "price":     price,
        "atr":       atr_val,
        "votes":     votes,
    }

def compute_levels(realtime_price, direction, atr_val):
    """
    Niveaux optimisés day trading :
      Zone d'entrée : ±0.3×ATR
      Stop Loss     : 2×ATR   (laisser respirer le trade)
      TP1           : 1.5×ATR (R:R 0.75 — sécuriser rapidement)
      TP2           : 3×ATR   (R:R 1.5)
      TP3           : 5×ATR   (R:R 2.5 — objectif ambitieux)
    """
    price      = realtime_price
    entry_half = round(atr_val * 0.3, 2)
    entry_low  = round(price - entry_half, 2)
    entry_high = round(price + entry_half, 2)
    sl_d       = round(atr_val * 2.0, 2)

    if direction == "BUY":
        sl  = round(price - sl_d, 2)
        tp1 = round(price + atr_val * 1.5, 2)
        tp2 = round(price + atr_val * 3.0, 2)
        tp3 = round(price + atr_val * 5.0, 2)
    else:
        sl  = round(price + sl_d, 2)
        tp1 = round(price - atr_val * 1.5, 2)
        tp2 = round(price - atr_val * 3.0, 2)
        tp3 = round(price - atr_val * 5.0, 2)

    return {
        "entry":      price,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "sl":         sl,
        "sl_dist":    sl_d,
        "tp1":  tp1, "rr1": round(1.5 / 2.0, 2),
        "tp2":  tp2, "rr2": round(3.0 / 2.0, 2),
        "tp3":  tp3, "rr3": round(5.0 / 2.0, 2),
    }

def signal_quality(s15m, s1h):
    combined = abs(s15m) + abs(s1h)
    if combined >= 9:  return "FORT",    "🔥🔥🔥"
    elif combined >= 7: return "ÉLEVÉ",  "🔥🔥"
    elif combined >= 5: return "MODÉRÉ", "🔥"
    else:               return "FAIBLE", "⚡"

# ═══════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        if resp.status_code == 200:
            log.info("Telegram envoyé ✓")
            return True
        log.error(f"Telegram erreur {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        log.error(f"Telegram exception: {e}")
        return False

def format_signal_message(direction, levels, a15m, a1h, qlabel, qemoji):
    now       = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    dir_emoji = "🟢" if direction == "BUY" else "🔴"
    dir_label = "ACHAT" if direction == "BUY" else "VENTE"
    entry     = levels["entry"]

    def pct(v):
        return f"{((v - entry) / entry * 100):+.3f}%"

    votes_lines = "".join(
        f"  {'🟢' if v > 0 else ('🔴' if v < 0 else '⚪')} {desc}\n"
        for _, (v, desc) in a15m["votes"].items()
    )

    return (
        f"╔══════════════════════════╗\n"
        f"║  🥇 SIGNAL XAU/USD Gold  ║\n"
        f"║     📅 DAY TRADING       ║\n"
        f"╚══════════════════════════╝\n\n"
        f"{dir_emoji} <b>{dir_label}</b> · Qualité: {qemoji} {qlabel}\n"
        f"⏱ Day Trading 15m (confirmé 1h)\n\n"
        f"━━━━━━━━━ NIVEAUX ━━━━━━━━━\n"
        f"💰 <b>Zone d'entrée :</b>\n"
        f"   ${levels['entry_low']:,.2f} — ${levels['entry_high']:,.2f}\n"
        f"   <i>(prix actuel : ${entry:,.2f})</i>\n\n"
        f"🛑 <b>Stop Loss :</b>  ${levels['sl']:,.2f}  "
        f"({pct(levels['sl'])} · -{levels['sl_dist']:.2f}$)\n\n"
        f"🎯 <b>TP1 :</b> ${levels['tp1']:,.2f}  "
        f"({pct(levels['tp1'])} · R:R {levels['rr1']})\n"
        f"🎯 <b>TP2 :</b> ${levels['tp2']:,.2f}  "
        f"({pct(levels['tp2'])} · R:R {levels['rr2']})\n"
        f"🎯 <b>TP3 :</b> ${levels['tp3']:,.2f}  "
        f"({pct(levels['tp3'])} · R:R {levels['rr3']})\n\n"
        f"━━━━ INDICATEURS 15m ━━━━━\n"
        f"{votes_lines}\n"
        f"Score 15m : {a15m['score']:+d}/6   Score 1h : {a1h['score']:+d}/6\n\n"
        f"⚠️ <i>Gérer votre risque. Signal automatique.</i>\n"
        f"🕐 {now}"
    )

# ═══════════════════════════════════════════════════
#  DONNÉES — Twelve Data
# ═══════════════════════════════════════════════════

def fetch_data(interval, bars):
    try:
        td_map = {"15m": "15min", "1h": "1h"}
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol=XAU/USD&interval={td_map[interval]}"
            f"&outputsize={bars}&apikey={TWELVE_DATA_KEY}"
        )
        resp  = requests.get(url, timeout=15)
        data  = resp.json()

        if data.get("status") == "error":
            log.error(f"Twelve Data erreur: {data.get('message')}")
            return None

        values = data.get("values")
        if not values or len(values) < 30:
            log.warning(f"Données insuffisantes {interval}: {len(values) if values else 0} barres")
            return None

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close", "volume": "Volume"
        })
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col])

        return df.tail(bars)

    except Exception as e:
        log.error(f"Erreur fetch {interval}: {e}")
        return None

# ═══════════════════════════════════════════════════
#  HEURES DE TRADING — 07h-21h Paris (sessions complètes)
# ═══════════════════════════════════════════════════

def is_trading_hours():
    """
    Day trading XAU/USD — sessions actives (heure de Paris UTC+2) :
      07h00 – 21h00  (pré-Londres + Londres + NY)
    Scan toutes les 15 min → 56 scans/jour × 2 timeframes = ~112 appels/jour.
    Largement sous la limite gratuite Twelve Data (800/jour).
    Fermé le week-end.
    """
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False

    paris_hour  = (now.hour + 2) % 24
    paris_min   = now.minute
    total_min   = paris_hour * 60 + paris_min

    return (7 * 60) <= total_min < (21 * 60)

def next_session_in():
    now        = datetime.now(timezone.utc)
    paris_hour = (now.hour + 2) % 24
    paris_min  = now.minute
    total_min  = paris_hour * 60 + paris_min

    if total_min < 7 * 60:
        return 7 * 60 - total_min
    return (24 * 60 - total_min) + 7 * 60  # Lendemain 07h

# ═══════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════

def run():
    log.info("━" * 50)
    log.info("  GOLD SIGNAL BOT v4.0 — Day Trading")
    log.info("━" * 50)

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("⛔ TELEGRAM_TOKEN et TELEGRAM_CHAT_ID doivent être définis.")
        return
    if not TWELVE_DATA_KEY:
        log.error("⛔ TWELVE_DATA_KEY doit être défini.")
        return

    send_telegram(
        "🤖 <b>Gold Signal Bot v4.0 — Day Trading</b>\n\n"
        f"📊 XAU/USD · 15m confirmé 1h\n"
        f"⏱ Scan toutes les {SCAN_INTERVAL_SEC // 60} min\n"
        f"🕐 Sessions : 07h–21h Paris\n"
        f"📡 BUY : {MIN_SCORE_15M}/6 (15m) + {MIN_SCORE_1H}/6 (1h)\n"
        f"📡 SELL : {MIN_SCORE_SELL_15M}/6 (15m) + {MIN_SCORE_SELL_1H}/6 (1h)\n"
        f"⏳ Cooldown : {COOLDOWN_MINUTES // 60}h entre signaux\n\n"
        f"✅ Bot opérationnel 24/7\n"
        f"🎯 Objectif : 1–3 trades/jour qualité"
    )

    last_signal_time = None
    scan_count = 0

    while True:
        try:
            scan_count += 1
            now = datetime.now(timezone.utc)
            log.info(f"Scan #{scan_count} — {now.strftime('%H:%M:%S UTC')}")

            # ── Heures de trading ──
            if not is_trading_hours():
                wait = next_session_in()
                log.info(f"  Hors session — prochaine dans {wait} min. Veille 5 min…")
                time.sleep(300)
                continue

            # ── Cooldown ──
            if last_signal_time:
                elapsed = (now - last_signal_time).total_seconds() / 60
                if elapsed < COOLDOWN_MINUTES:
                    remaining = COOLDOWN_MINUTES - elapsed
                    log.info(f"  Cooldown actif — encore {remaining:.0f} min")
                    time.sleep(SCAN_INTERVAL_SEC)
                    continue

            # ── Données OHLC ──
            df_15m = fetch_data("15m", BARS_15M)
            df_1h  = fetch_data("1h",  BARS_1H)

            if df_15m is None or df_1h is None:
                log.warning("  Données manquantes, retry dans 2 min…")
                time.sleep(120)
                continue

            # ── Indicateurs ──
            a15m = compute_signals(df_15m, "15m")
            a1h  = compute_signals(df_1h,  "1h")

            # ── Prix temps réel ──
            realtime_price = get_realtime_price()
            if realtime_price is None:
                realtime_price = a15m["price"]
                log.warning(f"  Fallback Twelve Data: ${realtime_price:,.2f}")

            log.info(
                f"  Prix: ${realtime_price:,.2f}  |  "
                f"Score 15m: {a15m['score']:+d}  |  Score 1h: {a1h['score']:+d}  |  "
                f"15m={a15m['direction']}  1h={a1h['direction']}"
            )

            # ── Détection signal ──
            # Confluences requises : 15m primaire + 1h confirme la direction
            direction = None

            if (a15m["direction"] == "BUY"
                    and a15m["score"] >= MIN_SCORE_15M
                    and a1h["score"] >= MIN_SCORE_1H):
                direction = "BUY"

            elif (a15m["direction"] == "SELL"
                    and a15m["score"] <= -MIN_SCORE_SELL_15M
                    and a1h["score"] <= -MIN_SCORE_SELL_1H):
                direction = "SELL"

            if direction:
                qlabel, qemoji = signal_quality(a15m["score"], a1h["score"])
                levels = compute_levels(realtime_price, direction, a15m["atr"])
                log.info(
                    f"  ✦ SIGNAL {direction} — {qlabel} — "
                    f"Zone ${levels['entry_low']:,.2f}–${levels['entry_high']:,.2f}  "
                    f"SL ${levels['sl']:,.2f}  TP3 ${levels['tp3']:,.2f}"
                )
                msg = format_signal_message(direction, levels, a15m, a1h, qlabel, qemoji)
                if send_telegram(msg):
                    last_signal_time = now
            else:
                log.info(f"  Pas de signal")

        except KeyboardInterrupt:
            log.info("Bot arrêté.")
            send_telegram("🔴 <b>Gold Signal Bot arrêté.</b>")
            break
        except Exception as e:
            log.error(f"Erreur: {e}")
            log.debug(traceback.format_exc())
            time.sleep(60)
            continue

        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    run()
