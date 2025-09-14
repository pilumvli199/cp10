
# main.py - Phase 4.2+ Breakouts (concise)
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT","TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL",1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[WARN] OpenAI init failed:", e)

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
OI_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
LONGSHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=30m&limit=1"

async def send_telegram(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"Markdown"}
    try:
        async with session.post(url, json=payload, timeout=15) as resp:
            if resp.status != 200:
                print("[ERROR] telegram send", await resp.text())
                return False
            return True
    except Exception as e:
        print("[ERROR] telegram exception", e)
        return False

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
    c30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='30m'))
    c1 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='1h'))
    c4 = await fetch_json(session, CANDLE_URL.format(symbol=symbol, interval='4h'))
    oi = await fetch_json(session, OI_URL.format(symbol=symbol))
    ls = await fetch_json(session, LONGSHORT_URL.format(symbol=symbol))
    out = {}
    if ticker:
        out['price'] = float(ticker.get('lastPrice',0))
        out['volume'] = float(ticker.get('volume',0))
    if isinstance(c30, list): out['candles_30m'] = [[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c30]
    if isinstance(c1, list): out['candles_1h'] = [[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c1]
    if isinstance(c4, list): out['candles_4h'] = [[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c4]
    if isinstance(oi, dict): out['oi'] = float(oi.get('openInterest',0))
    if isinstance(ls, list) and ls: out['long_short_ratio'] = float(ls[0].get('longShortRatio',1))
    return out

def calc_levels(candles, lookback=24, k=3):
    if not candles: return (None,None,None,None,None)
    arr = candles[-lookback:] if len(candles)>=lookback else candles[:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(k, max(1,len(highs)))
    res = sum(highs[:k])/k
    sup = sum(lows[:k])/k
    mid = (res+sup)/2 if res and sup else None
    r2 = sum(highs[k:k*2])/k if len(highs)>k else None
    s2 = sum(lows[k:k*2])/k if len(lows)>k else None
    return (sup,res,mid,r2,s2)

def detect_breaks(curr, prev):
    alerts = []
    if not curr: return alerts
    price = curr.get('price')
    candles = curr.get('candles_30m') or []
    if not price or not candles: return alerts
    sup,res,mid,r2,s2 = calc_levels(candles,24,3)
    prev_price = prev.get('price') if prev else None
    if res and prev_price is not None and prev_price<=res and price>res:
        alerts.append(f'Breakout ↑ above {res:.2f}')
    if sup and prev_price is not None and prev_price>=sup and price<sup:
        alerts.append(f'Breakdown ↓ below {sup:.2f}')
    if mid and prev_price is not None and prev_price<mid and price>mid:
        alerts.append(f'Crossed above mid {mid:.2f}')
    return alerts

async def analyze_openai(market_map):
    if not client: return None
    parts = []
    for s in SYMBOLS:
        d = market_map.get(s) or {}
        parts.append(f"{s}: price={d.get('price','NA')} vol={d.get('volume','NA')} oi={d.get('oi','NA')}")
        for tf in ('candles_30m','candles_1h','candles_4h'):
            if d.get(tf):
                last10 = d[tf][-10:]
                ct = ','.join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10])
                parts.append(f"{s} {tf} last10: {ct}")
    prompt = "You are concise. Output lines: SYMBOL - BIAS - TIMEFRAMES - REASON\n" + '\n'.join(parts)
    try:
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(model=OPENAI_MODEL, messages=[{'role':'user','content':prompt}], max_tokens=700, temperature=0.15))
        text = resp.choices[0].message.content.strip()
        print('=== OpenAI analysis ==='); print(text[:2000])
        return text
    except Exception as e:
        print('OpenAI error', e)
        return None

def parse_structured(text):
    out = {}
    if not text: return out
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = [p.strip() for p in ln.split(' - ')]
        if len(parts)>=3:
            sym = parts[0].upper(); bias=parts[1].upper(); tfs=parts[2]; reason=parts[3] if len(parts)>3 else ''
            out[sym] = {'bias':bias,'tfs':tfs,'reason':reason}
    return out

def detect_risk(prev,curr):
    a = []
    if not prev or not curr: return a
    try:
        pv=prev.get('volume'); cv=curr.get('volume')
        if pv and cv and cv>1.2*pv: a.append('⚠️ Volume spike >20%')
    except: pass
    try:
        poi=prev.get('oi'); coi=curr.get('oi')
        if poi and coi and coi>1.2*poi: a.append('⚠️ OI Jump >20%')
    except: pass
    return a

async def task_loop():
    async with aiohttp.ClientSession() as session:
        await send_telegram(session, '*Bot online — Phase 4.2+ Breakouts (alerts only)*')
        prev_map = {}; last_hist = {s:[] for s in SYMBOLS}; prev_break = {s:{'above':False,'below':False,'mid':False} for s in SYMBOLS}
        while True:
            start = time.time()
            tasks = [fetch_data(session,s) for s in SYMBOLS]
            res = await asyncio.gather(*tasks, return_exceptions=True)
            market_map = {}
            for i,s in enumerate(SYMBOLS):
                r = res[i]
                market_map[s] = r if not isinstance(r, Exception) and r else {}
            analysis = await analyze_openai(market_map)
            parsed = parse_structured(analysis) if analysis else {}
            # local breakouts
            for s in SYMBOLS:
                b_alerts = detect_breaks(market_map.get(s), prev_map.get(s))
                if b_alerts:
                    text = '; '.join(b_alerts)
                    # breakout up
                    if any('Breakout' in x for x in b_alerts) and not prev_break[s]['above']:
                        await send_telegram(session, f'🚨 ALERT: {s} → BREAKOUT · TF: 30m · Reason: {text}')
                        prev_break[s]['above']=True; prev_break[s]['below']=False
                    if any('Breakdown' in x for x in b_alerts) and not prev_break[s]['below']:
                        await send_telegram(session, f'🚨 ALERT: {s} → BREAKDOWN · TF: 30m · Reason: {text}')
                        prev_break[s]['below']=True; prev_break[s]['above']=False
                    if any('mid' in x.lower() for x in b_alerts) and not prev_break[s]['mid']:
                        await send_telegram(session, f'🚨 ALERT: {s} → MID-CROSS · TF: 30m · Reason: {text}')
                        prev_break[s]['mid']=True
                # reset if back inside
                curr = market_map.get(s) or {}
                debug_sup_res = calc_levels(curr.get('candles_30m') or [],24,3) if curr else (None,None,None,None,None)
                sup, res = debug_sup_res[0], debug_sup_res[1]
                p = curr.get('price')
                if p and sup and res and p<res and p>sup:
                    prev_break[s] = {'above':False,'below':False,'mid':False}
            # GPT alerts + strong
            for s,info in parsed.items():
                bias = info.get('bias','NEUTRAL'); tfs=info.get('tfs',''); reason=info.get('reason','')
                if bias in ('BUY','SELL'):
                    tfp = f' · TF: {tfs}' if tfs else ''
                    rp = f' · Reason: {reason}' if reason else ''
                    await send_telegram(session, f'🚨 ALERT: {s} → {bias}{tfp}{rp}')
                hist = last_hist.get(s,[]); hist.append(bias)
                if len(hist)>3: hist.pop(0)
                last_hist[s]=hist
                if len(hist)==3 and all(h==bias for h in hist) and bias in ('BUY','SELL'):
                    tfp = f' · TF: {tfs}' if tfs else ''
                    rp = f' · Reason: {reason}' if reason else ''
                    await send_telegram(session, f'🔥 STRONG {bias} Signal: {s} (3 confirmations){tfp}{rp}')
            # risk alerts
            for s in SYMBOLS:
                r = detect_risk(prev_map.get(s), market_map.get(s))
                for a in r:
                    await send_telegram(session, f'{a} on {s}')
            prev_map = market_map
            print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] cycle done. Parsed: {list(parsed.keys())}")
            elapsed = time.time()-start
            await asyncio.sleep(max(0, POLL_INTERVAL-elapsed))

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('[ERROR] set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env')
        return
    print('[INFO] Starting bot. Poll interval:', POLL_INTERVAL)
    try:
        asyncio.run(task_loop())
    except KeyboardInterrupt:
        print('Interrupted')

if __name__=='__main__':
    main()
