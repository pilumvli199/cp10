# main.py – Phase 4.3 (Analysis + Alerts)
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
OI_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
LONGSHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=30m&limit=1"

# ---------------- Telegram ----------------
async def send_telegram(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram creds missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    async with session.post(url, json=payload) as r:
        if r.status != 200:
            print("[ERROR] Telegram send failed:", await r.text())

# ---------------- Fetch ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                return None
            return await r.json()
    except:
        return None

async def fetch_data(session, symbol):
    ticker = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    c30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="30m"))
    c1h = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="1h"))
    c4h = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval="4h"))
    oi = await fetch_json(session, OI_URL.format(symbol=symbol))
    ls = await fetch_json(session, LONGSHORT_URL.format(symbol=symbol))
    out = {}
    if ticker:
        out["price"] = float(ticker.get("lastPrice", 0))
        out["volume"] = float(ticker.get("volume", 0))
    if isinstance(c30, list):
        out["candles_30m"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c30]
    if isinstance(c1h, list):
        out["candles_1h"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c1h]
    if isinstance(c4h, list):
        out["candles_4h"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in c4h]
    if isinstance(oi, dict):
        out["oi"] = float(oi.get("openInterest", 0))
    if isinstance(ls, list) and ls:
        out["long_short_ratio"] = float(ls[0].get("longShortRatio", 1))
    return out

# ---------------- OpenAI Analysis ----------------
async def analyze_openai(market_map):
    if not client:
        return None
    parts = []
    for s in SYMBOLS:
        d = market_map.get(s) or {}
        parts.append(f"{s}: price={d.get('price','NA')} vol={d.get('volume','NA')} oi={d.get('oi','NA')} ls={d.get('long_short_ratio','NA')}")
        for tf in ("candles_30m","candles_1h","candles_4h"):
            if d.get(tf):
                last10 = d[tf][-10:]
                ct = ",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10])
                parts.append(f"{s} {tf} last10: {ct}")
    prompt = (
        "You are a concise crypto technical analyst. For each symbol, output ONE LINE exactly:\n"
        "SYMBOL - BIAS - TIMEFRAMES - REASON\n"
        "Example:\nBTCUSDT - BUY - 30m,1h - ascending triangle breakout with rising volume\n\n"
        "Now analyze:\n" + "\n".join(parts)
    )
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.15,
        ),
    )
    return resp.choices[0].message.content.strip()

def parse_structured(text):
    out = {}
    if not text: return out
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split(" - ")]
        if len(parts) >= 3:
            sym = parts[0].upper()
            bias = parts[1].upper()
            tfs = parts[2]
            reason = parts[3] if len(parts) > 3 else ""
            out[sym] = {"bias": bias, "tfs": tfs, "reason": reason}
    return out

# ---------------- Risk detection ----------------
def detect_risk(prev,curr):
    a = []
    if not prev or not curr: return a
    try:
        pv,cv = prev.get("volume"), curr.get("volume")
        if pv and cv and cv > 1.2*pv: a.append("⚠️ Volume spike >20%")
    except: pass
    try:
        poi,coi = prev.get("oi"), curr.get("oi")
        if poi and coi and coi > 1.2*poi: a.append("⚠️ OI Jump >20%")
    except: pass
    return a

# ---------------- Main loop ----------------
async def task_loop():
    async with aiohttp.ClientSession() as session:
        await send_telegram(session, "*Bot online — Phase 4.3 (Analysis + Alerts)*")
        prev_map = {}; hist = {s: [] for s in SYMBOLS}
        while True:
            start = time.time()
            tasks = [fetch_data(session, s) for s in SYMBOLS]
            res = await asyncio.gather(*tasks)
            market_map = {s: r if r else {} for s,r in zip(SYMBOLS,res)}

            # 1) Get GPT analysis text
            analysis = await analyze_openai(market_map)
            parsed = parse_structured(analysis)

            # 2) Send normal analysis report (once per cycle)
            if analysis:
                report = "🧠 *Analysis Report* (UTC {})\n```\n{}\n```".format(
                    datetime.utcnow().strftime("%H:%M"), analysis
                )
                await send_telegram(session, report)

            # 3) Alerts (immediate, strong, risk)
            for s,info in parsed.items():
                bias = info.get("bias","NEUTRAL"); tfs=info.get("tfs",""); reason=info.get("reason","")
                if bias in ("BUY","SELL"):
                    await send_telegram(session, f"🚨 ALERT: {s} → {bias} · TF: {tfs} · Reason: {reason}")
                hist[s].append(bias)
                if len(hist[s]) > 3: hist[s].pop(0)
                if len(hist[s]) == 3 and all(x==bias for x in hist[s]) and bias in ("BUY","SELL"):
                    await send_telegram(session, f"🔥 STRONG {bias}: {s} (3 confirmations)")

            # 4) Risk alerts
            for s in SYMBOLS:
                r = detect_risk(prev_map.get(s), market_map.get(s))
                for msg in r:
                    await send_telegram(session, f"{msg} on {s}")

            prev_map = market_map
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] cycle done.")
            elapsed = time.time()-start
            await asyncio.sleep(max(0,POLL_INTERVAL-elapsed))

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram vars missing")
        return
    asyncio.run(task_loop())

if __name__=="__main__":
    main()
