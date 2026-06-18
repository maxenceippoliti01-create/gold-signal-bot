"""
╔══════════════════════════════════════════════════════════════╗
║       GOLD SIGNAL BOT v3.0 — Render/Railway (cloud 24/7)    ║
║  Lit les credentials depuis les variables d'environnement    ║
╚══════════════════════════════════════════════════════════════╝

Variables d'environnement requises :
  TELEGRAM_TOKEN   — token de ton bot BotFather
  TELEGRAM_CHAT_ID — ton chat ID Telegram
  TWELVE_DATA_KEY  — clé API Twelve Data (twelvedata.com gratuit)

Variables optionnelles :
  SCAN_INTERVAL      — secondes entre chaque scan (défaut: 60)
  COOLDOWN_MINUTES   — délai min entre 2 signaux (défaut: 15)
  MIN_SCORE_1M       — confluence BUY requise 1m (défaut: 4)
  MIN_SCORE_5M       — confluence BUY requise 5m (défaut: 2)
  MIN_SCORE_SELL_1M  — confluence SELL requise 1m (défaut: 3)
  MIN_SCORE_SELL_5M  — confluence SELL requise 5m (défaut: 2)

v3.0 — Corrections :
  - Zone d'entrée (low–high) au lieu d'un prix unique
  - Prix temps réel via Finnhub (corrige décalage $20–30 vs MT4/TradingView)
  - Seuil SELL à 3/6 (était 4/6) → signaux de vente activés
  - EMA50 neutre quand prix très proche (±0.15%) → évite biais haussier permanent
  - MACD neutre quand momentum s'inverse → moins de faux signaux
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
#  CONFIGURATION — depuis les variables d'environnement
# ═══════════════════════════════════════════════════

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY", "d8k2b71r01qjgd6qig80d8k2b71r01qjgd6qig8g")

SCAN_INTERVAL_SEC  = int(os.environ.get("SCAN_INTERVAL", "60"))
COOLDOWN_MINUTES   = int(os.environ.get("COOLDOWN_MINUTES", "15"))

# Seuils BUY (strict) et SELL (plus souple — l'or est souvent en tendance haussière)
MIN_SCORE_1M       = int(os.environ.get("MIN_SCORE_1M", "4"))
MIN_SCORE_5M       = int(os.environ.get("MIN_SCORE_5M", "2"))
MIN_SCORE_SELL_1M  = int(os.environ.get("MIN_SCORE_SELL_1M", "3"))
MIN_SCORE_SELL_5M  = int(os.environ.get("MIN_SCORE_SELL_5M", "2"))

BARS_1M = 120
BARS_5M = 60

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
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

def stochastic(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
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
#  PRIX TEMPS RÉEL — Finnhub (corrige décalage vs MT4)
# ═══════════════════════════════════════════════════

def get_realtime_price():
    """
    Récupère le prix temps réel de l'or via Finnhub.

    Pourquoi : Twelve Data retourne le CLOSE de la dernière bougie TERMINÉE.
    Sur 1m, cela signifie un retard de 0–60 secondes, mais avec la latence API
    et le fait que la bougie précédente est souvent la plus récente, on peut avoir
    1 à 5 minutes de décalage → $20–30 d'écart vs TradingView/MT4 en temps réel.
    Finnhub /quote renvoie le prix du marché actuel, sans délai.
    """
    try:
        resp = requests.get(
            f"https://finnhub.io/api/v1/quote?symbol=OANDA:XAU_USD&token={FINNHUB_KEY}",
            timeout=5
        )
        data = resp.json()
        price = data.get("c")  # "c" = current price
        if price and float(price) > 1000:  # Sanity check (or toujours > $1000)
            log.info(f"  Prix temps réel (Finnhub): ${float(price):,.2f}")
            return float(price)
        log.warning(f"  Finnhub quote inattendu: {data}")
        return None
    except Exception as e:
        log.warning(f"  Finnhub quote erreur: {e} — fallback Twelve Data")
        return None

# ═══════════════════════════════════════════════════
#  ANALYSE & SCORING
# ═══════════════════════════════════════════════════

def compute_signals(df):
    close, high, low = df["Close"], df["High"], df["Low"]

    rsi_val = rsi(close).iloc[-1]
    macd_line, signal_line, histogram = macd(close)
    hist_val  = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2]
    ema9  = ema(close, 9).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    bb_u = bb_upper.iloc[-1]
    bb_l = bb_lower.iloc[-1]
    bb_m = bb_mid.iloc[-1]
    price = close.iloc[-1]
    stoch_k, stoch_d = stochastic(high, low, close)
    sk = stoch_k.iloc[-1]; sd = stoch_d.iloc[-1]
    sk_prev = stoch_k.iloc[-2]; sd_prev = stoch_d.iloc[-2]
    atr_val = atr(high, low, close).iloc[-1]

    votes = {}

    # ── RSI ──
    if rsi_val < 35:   votes["RSI"] = ( 1, f"RSI {rsi_val:.1f} — Survendu ✅")
    elif rsi_val > 65: votes["RSI"] = (-1, f"RSI {rsi_val:.1f} — Suracheté ✅")
    else:              votes["RSI"] = ( 0, f"RSI {rsi_val:.1f} — Neutre")

    # ── MACD ──
    # Croisements = signal fort | histogramme qui s'inverse = neutre (momentum change)
    if hist_prev < 0 and hist_val >= 0:
        votes["MACD"] = ( 1, f"MACD croisement haussier ✅")
    elif hist_prev > 0 and hist_val <= 0:
        votes["MACD"] = (-1, f"MACD croisement baissier ✅")
    elif hist_val > 0 and hist_val < hist_prev:
        votes["MACD"] = ( 0, f"MACD haussier mais momentum baissier ({hist_val:.4f})")
    elif hist_val < 0 and hist_val > hist_prev:
        votes["MACD"] = ( 0, f"MACD baissier mais momentum haussier ({hist_val:.4f})")
    elif hist_val > 0:
        votes["MACD"] = ( 1, f"MACD haussier ({hist_val:.4f})")
    else:
        votes["MACD"] = (-1, f"MACD baissier ({hist_val:.4f})")

    # ── EMA 9/21 ──
    if ema9 > ema21:   votes["EMA9/21"] = ( 1, f"EMA9 > EMA21 ✅")
    else:              votes["EMA9/21"] = (-1, f"EMA9 < EMA21 ✅")

    # ── Bollinger Bands ──
    bb_range = bb_u - bb_l
    bb_pct = (price - bb_l) / bb_range if bb_range > 0 else 0.5
    if bb_pct < 0.2:   votes["BB"] = ( 1, f"Prix proche bande basse BB ✅")
    elif bb_pct > 0.8: votes["BB"] = (-1, f"Prix proche bande haute BB ✅")
    elif price > bb_m: votes["BB"] = ( 1, f"Prix au-dessus BB mid")
    else:              votes["BB"] = (-1, f"Prix sous BB mid")

    # ── Stochastique ──
    if sk < 20 and sk > sd and sk_prev <= sd_prev:
        votes["Stoch"] = ( 1, f"Stoch croisement haussier survendu ✅")
    elif sk > 80 and sk < sd and sk_prev >= sd_prev:
        votes["Stoch"] = (-1, f"Stoch croisement baissier suracheté ✅")
    elif sk < 30: votes["Stoch"] = ( 1, f"Stoch survendu ({sk:.1f})")
    elif sk > 70: votes["Stoch"] = (-1, f"Stoch suracheté ({sk:.1f})")
    else:         votes["Stoch"] = ( 0, f"Stoch neutre ({sk:.1f})")

    # ── EMA 50 (Tendance de fond) ──
    # Zone neutre si prix très proche de l'EMA50 (±0.15%) → évite de toujours voter +1
    diff_pct = (price - ema50) / ema50 * 100
    if abs(diff_pct) < 0.15:
        votes["EMA50"] = ( 0, f"Prix ≈ EMA50 (diff {diff_pct:+.2f}%) — Zone neutre")
    elif price > ema50:
        votes["EMA50"] = ( 1, f"Prix > EMA50 — Tendance haussière ✅")
    else:
        votes["EMA50"] = (-1, f"Prix < EMA50 — Tendance baissière ✅")

    score = sum(v[0] for v in votes.values())

    # Direction avec seuils asymétriques (SELL plus facile à atteindre)
    if score >= MIN_SCORE_1M:
        direction = "BUY"
    elif score <= -MIN_SCORE_SELL_1M:
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
    Calcule les niveaux avec une ZONE d'entrée (low–high) ± 0.2×ATR.
    Utilise le prix temps réel Finnhub pour être aligné avec MT4/TradingView.
    """
    price = realtime_price
    entry_half = round(atr_val * 0.2, 2)
    entry_low  = round(price - entry_half, 2)
    entry_high = round(price + entry_half, 2)
    sl_d       = round(atr_val * 1.5, 2)

    if direction == "BUY":
        sl  = round(price - sl_d, 2)
        tp1 = round(price + atr_val * 1.0, 2)
        tp2 = round(price + atr_val * 2.0, 2)
        tp3 = round(price + atr_val * 3.0, 2)
    else:  # SELL
        sl  = round(price + sl_d, 2)
        tp1 = round(price - atr_val * 1.0, 2)
        tp2 = round(price - atr_val * 2.0, 2)
        tp3 = round(price - atr_val * 3.0, 2)

    return {
        "entry":      price,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "sl":         sl,
        "sl_dist":    sl_d,
        "tp1":  tp1, "rr1": round(atr_val / sl_d, 2),
        "tp2":  tp2, "rr2": round(atr_val * 2 / sl_d, 2),
        "tp3":  tp3, "rr3": round(atr_val * 3 / sl_d, 2),
    }

def signal_quality(s1, s5):
    combined = abs(s1) + abs(s5)
    if combined >= 10: return "FORT",    "🔥🔥🔥"
    elif combined >= 8: return "ÉLEVÉ",  "🔥🔥"
    elif combined >= 6: return "MODÉRÉ", "🔥"
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

def format_signal_message(direction, levels, a1m, a5m, qlabel, qemoji):
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    dir_emoji = "🟢" if direction == "BUY" else "🔴"
    dir_label = "ACHAT" if direction == "BUY" else "VENTE"
    entry = levels["entry"]

    def pct(v):
        return f"{((v - entry) / entry * 100):+.2f}%"

    votes_lines = "".join(
        f"  {'🟢' if v > 0 else ('🔴' if v < 0 else '⚪')} {desc}\n"
        for _, (v, desc) in a1m["votes"].items()
    )

    return (
        f"╔══════════════════════════╗\n"
        f"║  🥇 SIGNAL XAU/USD Gold  ║\n"
        f"╚══════════════════════════╝\n\n"
        f"{dir_emoji} <b>{dir_label}</b> · Qualité: {qemoji} {qlabel}\n"
        f"⏱ Scalping 1m (confirmé 5m)\n\n"
        f"━━━━━━━━━ NIVEAUX ━━━━━━━━━\n"
        f"💰 <b>Zone d'entrée :</b>\n"
        f"   ${levels['entry_low']:,.2f} — ${levels['entry_high']:,.2f}\n"
        f"   <i>(prix actuel : ${entry:,.2f})</i>\n\n"
        f"🛑 <b>Stop Loss :</b>  ${levels['sl']:,.2f}  "
        f"({pct(levels['sl'])} · -{levels['sl_dist']:.2f}$)\n"
        f"🎯 <b>TP1 :</b>       ${levels['tp1']:,.2f}  "
        f"({pct(levels['tp1'])} · R:R {levels['rr1']})\n"
        f"🎯 <b>TP2 :</b>       ${levels['tp2']:,.2f}  "
        f"({pct(levels['tp2'])} · R:R {levels['rr2']})\n"
        f"🎯 <b>TP3 :</b>       ${levels['tp3']:,.2f}  "
        f"({pct(levels['tp3'])} · R:R {levels['rr3']})\n\n"
        f"━━━━ INDICATEURS 1m ━━━━━\n"
        f"{votes_lines}\n"
        f"Score 1m : {a1m['score']:+d}/6   Score 5m : {a5m['score']:+d}/6\n\n"
        f"⚠️ <i>Gérer votre risque. Signal automatique.</i>\n"
        f"🕐 {now}"
    )

# ═══════════════════════════════════════════════════
#  DONNÉES — Twelve Data (OHLC pour les indicateurs)
# ═══════════════════════════════════════════════════

def fetch_data(symbol, interval, bars):
    """
    Récupère les bougies OHLC depuis Twelve Data pour calculer les indicateurs.
    Le prix affiché dans les signaux vient de Finnhub (temps réel), pas de Twelve Data.
    """
    try:
        td_interval = "1min" if interval == "1m" else "5min"
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol=XAU/USD&interval={td_interval}"
            f"&outputsize={bars}&apikey={TWELVE_DATA_KEY}"
        )
        resp = requests.get(url, timeout=15)
        data = resp.json()

        if data.get("status") == "error":
            log.error(f"Twelve Data erreur: {data.get('message')}")
            return None

        values = data.get("values")
        if not values or len(values) < 60:
            log.warning(f"Twelve Data données insuffisantes {interval}: "
                        f"{len(values) if values else 0} barres")
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
#  HEURES DE TRADING (Paris = UTC+2 en été)
# ═══════════════════════════════════════════════════

def is_trading_hours():
    """
    Créneaux actifs (heure de Paris) :
      Matin     : 08h00 – 12h00
      Après-midi: 14h00 – 16h40
    = 800 appels/jour exactement (limite free Twelve Data).
    Fermé le week-end.
    """
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False

    paris_hour   = (now.hour + 2) % 24  # UTC+2 (CEST, avril–octobre)
    paris_minute = now.minute
    total_min    = paris_hour * 60 + paris_minute

    in_morning   = (8 * 60) <= total_min < (12 * 60)
    in_afternoon = (14 * 60) <= total_min < (16 * 60 + 40)
    return in_morning or in_afternoon

def next_session_in():
    now = datetime.now(timezone.utc)
    paris_hour   = (now.hour + 2) % 24
    paris_minute = now.minute
    total_min    = paris_hour * 60 + paris_minute

    for s in [8 * 60, 14 * 60]:
        if total_min < s:
            return s - total_min
    return (24 * 60 - total_min) + 8 * 60

# ═══════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════

def run():
    log.info("━" * 50)
    log.info("  GOLD SIGNAL BOT v3.0 — Render Edition")
    log.info("━" * 50)

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("⛔ TELEGRAM_TOKEN et TELEGRAM_CHAT_ID doivent être définis.")
        return
    if not TWELVE_DATA_KEY:
        log.error("⛔ TWELVE_DATA_KEY doit être défini (compte gratuit twelvedata.com).")
        return

    send_telegram(
        "🤖 <b>Gold Signal Bot v3.0 démarré</b>\n\n"
        f"📊 XAU/USD (prix temps réel Finnhub)\n"
        f"⏱ Scan toutes les {SCAN_INTERVAL_SEC}s\n"
        f"📡 BUY : {MIN_SCORE_1M}/6 (1m) + {MIN_SCORE_5M}/6 (5m)\n"
        f"📡 SELL : {MIN_SCORE_SELL_1M}/6 (1m) + {MIN_SCORE_SELL_5M}/6 (5m)\n"
        f"⏳ Cooldown : {COOLDOWN_MINUTES} min\n\n"
        f"✅ Bot opérationnel 24/7\n"
        f"🆕 Zone d'entrée · Prix corrigé · Signaux SELL activés"
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
                log.info(f"  Hors session — prochaine dans {wait} min. Veille 60s…")
                time.sleep(60)
                continue

            # ── Cooldown ──
            if last_signal_time:
                elapsed = (now - last_signal_time).total_seconds() / 60
                if elapsed < COOLDOWN_MINUTES:
                    log.info(f"  Cooldown actif ({elapsed:.1f}/{COOLDOWN_MINUTES} min)")
                    time.sleep(SCAN_INTERVAL_SEC)
                    continue

            # ── Données OHLC ──
            df_1m = fetch_data("XAU/USD", "1m", BARS_1M)
            df_5m = fetch_data("XAU/USD", "5m", BARS_5M)

            if df_1m is None or df_5m is None:
                log.warning("  Données manquantes, retry dans 30s…")
                time.sleep(30)
                continue

            # ── Indicateurs ──
            a1m = compute_signals(df_1m)
            a5m = compute_signals(df_5m)

            # ── Prix temps réel (Finnhub) ──
            realtime_price = get_realtime_price()
            if realtime_price is None:
                realtime_price = a1m["price"]  # Fallback Twelve Data
                log.warning(f"  Fallback prix Twelve Data: ${realtime_price:,.2f}")

            log.info(
                f"  Prix: ${realtime_price:,.2f}  |  "
                f"Score 1m: {a1m['score']:+d}  |  Score 5m: {a5m['score']:+d}"
            )

            # ── Détection signal ──
            direction = None
            if a1m["direction"] == "BUY" \
                    and a1m["score"] >= MIN_SCORE_1M \
                    and a5m["score"] >= MIN_SCORE_5M:
                direction = "BUY"
            elif a1m["direction"] == "SELL" \
                    and a1m["score"] <= -MIN_SCORE_SELL_1M \
                    and a5m["score"] <= -MIN_SCORE_SELL_5M:
                direction = "SELL"

            if direction:
                qlabel, qemoji = signal_quality(a1m["score"], a5m["score"])
                levels = compute_levels(realtime_price, direction, a1m["atr"])
                log.info(
                    f"  ✦ SIGNAL {direction} — {qlabel} — "
                    f"Zone ${levels['entry_low']:,.2f}–${levels['entry_high']:,.2f}"
                )
                msg = format_signal_message(direction, levels, a1m, a5m, qlabel, qemoji)
                if send_telegram(msg):
                    last_signal_time = now
            else:
                log.info(
                    f"  Pas de signal — "
                    f"1m={a1m['direction']} ({a1m['score']:+d}), "
                    f"5m={a5m['direction']} ({a5m['score']:+d})"
                )

        except KeyboardInterrupt:
            log.info("Bot arrêté.")
            send_telegram("🔴 <b>Gold Signal Bot arrêté.</b>")
            break
        except Exception as e:
            log.error(f"Erreur: {e}")
            log.debug(traceback.format_exc())
            time.sleep(30)
            continue

        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    run()
