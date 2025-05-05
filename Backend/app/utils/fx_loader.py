import os
import requests
from datetime import datetime

from Shared.config.dotenv_loader import load_environment
from Backend.app.models.data_models import Currency, ExchangeRate

def fetch_full_fx_history(from_symbol: str, to_symbol: str) -> dict:
    """
    Retrieves the full daily FX history for a currency pair from Alpha Vantage.
    Returns the raw JSON which includes 'Meta Data' and 'Time Series FX (Daily)'.
    """
    load_environment()  # ensures ALPHAVANTAGE_API_KEY is set
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = (
        "https://www.alphavantage.co/query?"
        f"function=FX_DAILY&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}&outputsize=full&apikey={api_key}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_intraday_fx_history(from_symbol: str,
                              to_symbol: str,
                              interval: str = "5min",
                              outputsize: str = "full") -> dict:
    """
    Retrieves intraday FX history for a currency pair from Alpha Vantage.
    
    :param from_symbol: Base currency code, e.g. "EUR"
    :param to_symbol:   Quote currency code, e.g. "USD"
    :param interval:    One of "1min", "5min", "15min", "30min", "60min"
    :param outputsize:  "compact" (latest 100 points) or "full" (up to 30 days)
    :returns:           Raw JSON including 'Meta Data' and
                       'Time Series FX (<interval>)'.
    """
    load_environment()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = (
        "https://www.alphavantage.co/query?"
        f"function=FX_INTRADAY&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}&interval={interval}"
        f"&outputsize={outputsize}&apikey={api_key}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def upsert_currencies(session, codes: list[str]):
    """
    Ensures each currency code exists in the Currency table.
    Inserts new Currency records for any codes not already present.
    """
    existing = {
        c.code
        for c in session.query(Currency)
                       .filter(Currency.code.in_(codes))
                       .all()
    }
    for code in codes:
        if code not in existing:
            session.add(Currency(code=code, name=code))
    session.commit()


def upsert_exchange_rates(session, base: str, quote: str, records: list[dict]):
    """
    Upserts exchange rate records into the ExchangeRate table.
    - Ensures base and quote currencies exist.
    - Inserts or updates OHLC and timestamp values.
    """
    upsert_currencies(session, [base, quote])
    base_obj = session.query(Currency).filter_by(code=base).one()
    quote_obj = session.query(Currency).filter_by(code=quote).one()

    for rec in records:
        ts = datetime.strptime(rec["date"], "%Y-%m-%d %H:%M:%S"
                               if " " in rec["date"] else "%Y-%m-%d")
        # support both daily and intraday keys
        open_val  = float(rec.get("1. open", rec.get("open")))
        high_val  = float(rec.get("2. high", rec.get("high")))
        low_val   = float(rec.get("3. low",  rec.get("low")))
        close_val = float(rec.get("4. close",rec.get("close")))

        existing = (
            session.query(ExchangeRate)
                   .filter_by(
                       base_currency_id=base_obj.id,
                       quote_currency_id=quote_obj.id,
                       timestamp=ts
                   )
                   .one_or_none()
        )

        if existing:
            existing.open  = open_val
            existing.high  = high_val
            existing.low   = low_val
            existing.close = close_val
        else:
            session.add(ExchangeRate(
                base_currency_id=base_obj.id,
                quote_currency_id=quote_obj.id,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close_val,
                timestamp=ts
            ))
    session.commit()
