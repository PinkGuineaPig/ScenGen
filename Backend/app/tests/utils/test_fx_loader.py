import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.app.models.data_models import Currency, ExchangeRate
from Backend.app.utils.fx_loader import (
    fetch_full_fx_history,
    fetch_intraday_fx_history,
    upsert_currencies,
    upsert_exchange_rates,
)

# Use in-memory SQLite for isolated, fast tests
TEST_DB_URI = "sqlite:///:memory:"

@pytest.fixture(scope="module")
def engine():
    """
    Create an in-memory SQLite engine once per test module.
    """
    return create_engine(TEST_DB_URI)

@pytest.fixture(scope="module")
def tables(engine):
    """
    Set up and tear down ONLY the Currency and ExchangeRate tables.
    This avoids hitting unsupported types (e.g. ARRAY) in SQLite.
    """
    # Create only the two tables we need
    Currency.__table__.create(engine)
    ExchangeRate.__table__.create(engine)
    yield
    # Drop only those two tables
    ExchangeRate.__table__.drop(engine)
    Currency.__table__.drop(engine)

@pytest.fixture()
def session(engine, tables):
    """
    Provide a transactional scope for each test. Rolls back on teardown.
    """
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.rollback()
    sess.close()

class DummyResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP {self.status_code}")

# --- FX fetch tests ---

def test_fetch_full_fx_history_requests_and_parses(monkeypatch):
    # Arrange: stub load_environment in the fx_loader module
    monkeypatch.setattr(
        "Backend.app.utils.fx_loader.load_environment",
        lambda: None
    )
    # Provide a fake API key in the environment
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TESTKEY123")

    fake_json = {
        "Meta Data": {
            "1. Information":    "Daily Foreign Exchange Rates",
            "2. From Symbol":    "EUR",
            "3. To Symbol":      "USD",
            "4. Output Size":    "Full size",
            "5. Last Refreshed": "2025-04-17 00:00:00",
            "6. Time Zone":      "UTC"
        },
        "Time Series FX (Daily)": {
            "2025-04-17": {"1. open":"1.0800", "2. high":"1.0850", "3. low":"1.0750", "4. close":"1.0820"},
            "2025-04-16": {"1. open":"1.0780", "2. high":"1.0830", "3. low":"1.0740", "4. close":"1.0800"},
        }
    }

    # Stub out the HTTP call in fx_loader
    def fake_get(url):
        assert "function=FX_DAILY"    in url
        assert "outputsize=full"      in url
        assert "apikey=TESTKEY123"     in url
        return DummyResponse(fake_json)

    monkeypatch.setattr(
        "Backend.app.utils.fx_loader.requests.get",
        fake_get
    )

    # Act
    result = fetch_full_fx_history("EUR", "USD")

    # Assert that the JSON was returned and parsed correctly
    ts = result["Time Series FX (Daily)"]
    assert ts["2025-04-17"]["4. close"] == "1.0820"
    assert ts["2025-04-16"]["1. open"]  == "1.0780"



def test_fetch_intraday_fx_history_requests_and_parses(monkeypatch):
    # Arrange
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TESTKEY123")

    # Arrange: stub load_environment in the fx_loader module
    monkeypatch.setattr(
        "Backend.app.utils.fx_loader.load_environment",
        lambda: None
    )

    fake_intraday = {
        "Time Series FX (5min)": {
            "2025-04-17 09:00:00": {"1. open":"1.0800","2. high":"1.0805","3. low":"1.0795","4. close":"1.0802"}
        }
    }

    def fake_get(url):
        assert "function=FX_INTRADAY" in url
        assert "interval=5min" in url
        assert "outputsize=full" in url
        assert "apikey=TESTKEY123" in url
        return DummyResponse(fake_intraday)

    monkeypatch.setattr("requests.get", fake_get)

    # Act
    data = fetch_intraday_fx_history("EUR", "USD", interval="5min")

    # Assert
    series = data["Time Series FX (5min)"]
    assert series["2025-04-17 09:00:00"]["1. open"] == "1.0800"
    assert series["2025-04-17 09:00:00"]["2. high"] == "1.0805"
    assert series["2025-04-17 09:00:00"]["3. low"] == "1.0795"
    assert series["2025-04-17 09:00:00"]["4. close"] == "1.0802"

# --- Upsert tests ---

def test_upsert_currencies(session):
    # Initially empty
    assert session.query(Currency).count() == 0

    # Insert new codes
    upsert_currencies(session, ["EUR", "USD"])
    codes = {c.code for c in session.query(Currency).all()}
    assert codes == {"EUR", "USD"}

    # Add one existing, one new
    upsert_currencies(session, ["USD", "JPY"])
    codes = {c.code for c in session.query(Currency).all()}
    assert codes == {"EUR", "USD", "JPY"}


def test_upsert_exchange_rates(session):
    # Ensure currencies
    upsert_currencies(session, ["EUR", "USD"])

    # Prepare daily records
    records = [
        {"date":"2025-04-17","open":"1.0800","high":"1.0850","low":"1.0750","close":"1.0820"},
        {"date":"2025-04-16","open":"1.0780","high":"1.0830","low":"1.0740","close":"1.0800"},
    ]
    upsert_exchange_rates(session, "EUR", "USD", records)
    all_rates = session.query(ExchangeRate).all()
    assert len(all_rates) == 2

    # Update first record
    updated = [{"date":"2025-04-17","open":"1.0790","high":"1.0840","low":"1.0740","close":"1.0810"}]
    upsert_exchange_rates(session, "EUR", "USD", updated)
    row = session.query(ExchangeRate).filter_by(timestamp=datetime(2025,4,17)).one()
    assert float(row.close) == 1.0810