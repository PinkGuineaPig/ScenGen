#!/usr/bin/env python
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
import io
import csv
from datetime import datetime

# Flask application setup to access the database
from Backend.app import create_app, db
from Backend.app.models.data_models import Currency, ExchangeRate

ROOT_URL = "https://www.histdata.com"
PAIRS_PATH = "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes"
PAIRS_URL = ROOT_URL + PAIRS_PATH
HEADERS = {"User-Agent": "HistData FX-Loader/1.0"}


def get_available_pairs():
    resp = requests.get(PAIRS_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='table') or soup.find('table', {'width': '500px'})
    if not table:
        print("⚠️ Could not find pairs table.")
        return []
    pairs = []
    for row in table.find_all('tr'):
        a = row.find('a', href=True)
        strong = a.find('strong') if a else None
        if not a or not strong:
            continue
        pair = strong.get_text(strip=True)
        code = pair.replace('/', '').lower()
        page_url = urljoin(ROOT_URL, f"{PAIRS_PATH}/{code}")
        pairs.append({'pair': pair, 'url': page_url})
    print(f"Found {len(pairs)} pairs.")
    return pairs


def upsert_currencies(session, codes):
    existing = {c.code for c in session.query(Currency).filter(Currency.code.in_(codes)).all()}
    for code in codes:
        if code not in existing:
            session.add(Currency(code=code, name=code))
    session.commit()


def upsert_exchange_rates(session, base, quote, records, interval='1min'):
    upsert_currencies(session, [base, quote])
    base_obj = session.query(Currency).filter_by(code=base).one_or_none()
    quote_obj = session.query(Currency).filter_by(code=quote).one_or_none()
    if not base_obj or not quote_obj:
        raise ValueError(f"Currency lookup failed: {base}/{quote}")
    print(f"→ Upserting {len(records)} records for {base}/{quote}")
    for rec in records:
        ts = datetime.strptime(rec['date'], "%Y-%m-%d %H:%M:%S")
        open_val, high_val = float(rec['open']), float(rec['high'])
        low_val, close_val = float(rec['low']), float(rec['close'])
        existing = session.query(ExchangeRate).filter_by(
            base_currency_id=base_obj.id,
            quote_currency_id=quote_obj.id,
            interval=interval,
            timestamp=ts
        ).one_or_none()
        if existing:
            existing.open, existing.high = open_val, high_val
            existing.low, existing.close = low_val, close_val
        else:
            session.add(ExchangeRate(
                base_currency_id=base_obj.id,
                quote_currency_id=quote_obj.id,
                interval=interval,
                open=open_val, high=high_val,
                low=low_val, close=close_val,
                timestamp=ts
            ))
    session.commit()


def load_pair_history_to_db(session, pair: str):
    entries = get_available_pairs()
    entry = next((e for e in entries if e['pair'] == pair), None)
    if not entry:
        print(f"⚠️ Pair {pair} not found; skipping.")
        return
    base, quote = pair.split('/')
    code = (base + quote).lower()
    print(f"→ Loading {pair} from {entry['url']}")
    resp = requests.get(entry['url'], headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'html.parser')
    link_prefix = f"{PAIRS_PATH}/{code}/"
    paths = {a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(link_prefix)}
    page_urls = sorted(urljoin(ROOT_URL, p) for p in paths)
    print(f"→ Found {len(page_urls)} pages for {pair}")

    records = []
    for page_url in page_urls:
        print(f"   - Visiting {page_url}")
        page_resp = requests.get(page_url, headers=HEADERS)
        page_resp.raise_for_status()
        page_soup = BeautifulSoup(page_resp.text, 'html.parser')

        form = page_soup.find('form', id='file_down')
        if not form:
            print(f"     ⚠️ No download form on {page_url}")
            continue
        action = form.get('action')
        form_url = urljoin(ROOT_URL, action)
        payload = {inp['name']: inp.get('value', '') for inp in form.find_all('input', type='hidden') if inp.get('name')}

        print(f"     • POST to {form_url} (referer: {page_url})")
        headers_post = HEADERS.copy()
        headers_post['Referer'] = page_url
        zresp = requests.post(form_url, data=payload, headers=headers_post)

        try:
            with zipfile.ZipFile(io.BytesIO(zresp.content)) as zf:
                for fname in zf.namelist():
                    if not fname.lower().endswith('.csv'):
                        continue
                    print(f"       Parsing CSV {fname}")
                    with zf.open(fname) as cf:
                        raw = cf.read().decode('utf-8', errors='ignore')
                        lines = raw.splitlines()
                        print(f"         Read {len(lines)} total lines")
                        data_lines = [ln for ln in lines if ln and not ln.lower().startswith('date')]
                        print(f"         {len(data_lines)} data lines after header filter")
                        print(f"         Sample data lines: {data_lines[:3]}")
                        reader = csv.reader(data_lines, delimiter=';')
                        count = 0
                        for i, row in enumerate(reader):
                            if i < 3:
                                print(f"           Row sample: {row}")
                            if len(row) < 5:
                                continue
                            # detect time format (HHMM or HHMMSS)
                            date_str = row[0]
                            parts = date_str.split(' ')
                            if len(parts) < 2:
                                print(f"           Skipping malformed timestamp: {date_str}")
                                continue
                            time_part = parts[1]
                            fmt = "%Y%m%d %H%M%S" if len(time_part) == 6 else "%Y%m%d %H%M"
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                records.append({
                                    'date': dt.strftime("%Y-%m-%d %H:%M:%S"),
                                    'open': row[1], 'high': row[2],
                                    'low': row[3], 'close': row[4]
                                })
                                count += 1
                            except Exception as e:
                                print(f"           Failed parsing row {row}: {e}")
                                continue
                        print(f"         Appended {count} records from {fname}")
        except zipfile.BadZipFile:
            print(f"       ⚠️ Skipped non-zip response from {form_url}")
            continue

    print(f"→ Parsed total {len(records)} records for {pair}")
    if records:
        upsert_exchange_rates(session, base, quote, records)
        print(f"✓ Upserted {len(records)} records for {pair}")


def main():
    parser = argparse.ArgumentParser(description="HistData FX Loader")
    parser.add_argument('--list', '-l', action='store_true', help='List pairs')
    parser.add_argument('--pair', '-p', type=str, help='Load specific pair')
    args = parser.parse_args()
    if args.list:
        for itm in get_available_pairs(): print(itm['pair'])
        sys.exit(0)

    app = create_app()
    with app.app_context():
        db.create_all()
        session = db.session
        targets = [args.pair] if args.pair else [e['pair'] for e in get_available_pairs()]
        for pair in targets:
            load_pair_history_to_db(session, pair)
        print("🎉 Done!")

if __name__ == '__main__':
    main()
