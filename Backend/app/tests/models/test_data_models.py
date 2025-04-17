import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.app.models.data_models import Currency, ExchangeRate

# Use an in-memory SQLite database for fast, isolated tests
TEST_DB_URI = "sqlite:///:memory:"

@pytest.fixture(scope="module")
def engine():
    return create_engine(TEST_DB_URI)

@pytest.fixture(scope="module")
def tables(engine):
    # Create tables for Currency and ExchangeRate
    Currency.metadata.create_all(engine)
    ExchangeRate.metadata.create_all(engine)
    yield
    # Drop tables after tests
    ExchangeRate.metadata.drop_all(engine)
    Currency.metadata.drop_all(engine)

@pytest.fixture()
def session(engine, tables):
    """
    Provides a transactional scope around a series of operations.
    Rolls back after each test for isolation.
    """
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.rollback()
    sess.close()

## just here to test that we start with a clean DB
def test_currency_table_empty(session):
    """Edge case: no currencies exist initially."""
    assert session.query(Currency).count() == 0


