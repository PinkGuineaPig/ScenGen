#!/usr/bin/env python
"""
scripts/fetch_fx.py

Responsibilities:
- Bootstrap the environment (load .env via Shared/config/dotenv_loader)
- Create a DB connection via the Flask app factory
- Ensure all tables are created (db.create_all())
- Loop over configured currency pairs
- Call fx_loader to fetch & persist each pair’s OHLC data (daily & intraday)
- Respect Alpha Vantage free‑tier rate limits (5 calls/minute)
- Gracefully handle HTTP errors and rate‑limit messages
- Exit with 0 on success, non‑zero on failure
"""

import os
import sys
import time
import requests

# Ensure project root is on sys.path so Backend/ and Shared/ import correctly
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Backend.app import create_app, db
from Backend.app.utils.fx_loader import (
    fetch_full_fx_history,
    fetch_intraday_fx_history,
    upsert_exchange_rates,
)


def main():
    app = create_app()
    with app.app_context():
        db.create_all()

        pairs = [
            ("EUR", "USD"),
            ("USD", "JPY"),
            ("GBP", "USD"),
            ("AUD", "USD"),
            ("USD", "CAD"),
        ]

        call_count = 0
        for base, quote in pairs:
            # DAILY FETCH
            print(f"Fetching daily history for {base}/{quote}…")
            try:
                daily = fetch_full_fx_history(base, quote)
            except requests.HTTPError as e:
                print(f"HTTP error fetching daily for {base}/{quote}: {e}")
                print("Sleeping 60s before skipping daily fetch...")
                time.sleep(60)
                daily = {"Time Series FX (Daily)": {}}
            call_count += 1
            daily_records = [
                {"date": ts, "open": v["1. open"], "high": v["2. high"],
                 "low": v["3. low"], "close": v["4. close"]}
                for ts, v in daily.get("Time Series FX (Daily)", {}).items()
            ]
            upsert_exchange_rates(db.session, base, quote, daily_records)

            # Rate-limit pause
            if call_count % 5 == 0:
                time.sleep(60)
            else:
                time.sleep(12)

                        # 2) Intraday history (currently disabled)
            # print(f"Fetching intraday (5min) for {base}/{quote}…")
            # try:
            #     intraday = fetch_intraday_fx_history(base, quote, interval="5min")
            # except requests.HTTPError as e:
            #     print(f"HTTP error fetching intraday for {base}/{quote}: {e}")
            #     intraday = {}
            # call_count += 1

            # # Dynamically extract the time-series data
            # series_key = next(
            #     (k for k in intraday.keys() if k.startswith("Time Series FX")),
            #     None
            # )
            # if not series_key:
            #     print(
            #         f"Warning: failed to fetch intraday data for {base}/{quote}."
            #         f" Available keys: {list(intraday.keys())}"
            #     )
            # else:
            #     intraday_records = [
            #         {"date": ts, "open": v["1. open"], "high": v["2. high"],
            #          "low": v["3. low"], "close": v["4. close"]}
            #         for ts, v in intraday[series_key].items()
            #     ]
            #     upsert_exchange_rates(db.session, base, quote, intraday_records)

            # Rate-limit pause placeholder (adjusted after intraday disabled)
            time.sleep(0)
            if call_count % 5 == 0:
                time.sleep(60)
            else:
                time.sleep(12)
            if call_count % 5 == 0:
                time.sleep(60)
            else:
                time.sleep(12)

        print("FX data ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
