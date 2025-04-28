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


def upsert_exchange_rates(
    session,
    base: str,
    quote: str,
    records: list[dict],
    interval: str = '1d'
):
    """
    Upserts exchange rate records into the ExchangeRate table.
    
    :param session:       SQLAlchemy session
    :param base:          Base currency code (e.g. "EUR")
    :param quote:         Quote currency code (e.g. "USD")
    :param records:       List of dicts with keys like "date", "open"/"1. open", etc.
    :param interval:      Bar size, e.g. "1d", "1min", "5min", etc.
    """
    # 1) Ensure currencies exist
    upsert_currencies(session, [base, quote])
    base_obj  = session.query(Currency).filter_by(code=base).one()
    quote_obj = session.query(Currency).filter_by(code=quote).one()

    # 2) Upsert each record
    for rec in records:
        # parse timestamp (supports "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS")
        date_str = rec["date"]
        fmt = "%Y-%m-%d %H:%M:%S" if " " in date_str else "%Y-%m-%d"
        ts = datetime.strptime(date_str, fmt)

        # extract OHLC values (supports both AV keys and generic keys)
        open_val  = float(rec.get("1. open", rec.get("open")))
        high_val  = float(rec.get("2. high", rec.get("high")))
        low_val   = float(rec.get("3. low",  rec.get("low")))
        close_val = float(rec.get("4. close", rec.get("close")))

        # check for existing row
        existing = (
            session.query(ExchangeRate)
                   .filter_by(
                       base_currency_id=base_obj.id,
                       quote_currency_id=quote_obj.id,
                       interval=interval,
                       timestamp=ts
                   )
                   .one_or_none()
        )

        if existing:
            # update values
            existing.open  = open_val
            existing.high  = high_val
            existing.low   = low_val
            existing.close = close_val
        else:
            # insert new row
            session.add(ExchangeRate(
                base_currency_id=base_obj.id,
                quote_currency_id=quote_obj.id,
                interval=interval,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close_val,
                timestamp=ts
            ))

    # 3) Commit all changes
    session.commit()
