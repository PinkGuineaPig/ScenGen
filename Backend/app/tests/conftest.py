# Backend/app/tests/conftest.py

import pytest
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import types as sqltypes
from Backend.app import create_app, db as _db

@pytest.fixture(scope="session")
def app():
    """
    Application configured for testing with an in-memory SQLite database.
    Patches any PostgreSQL ARRAY columns (regardless of origin) to JSON for SQLite compatibility,
    then creates the full schema.
    """
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })

    with app.app_context():
        # Ensure all models are imported so their tables register in metadata
        import Backend.app.models.run_models       # registers ModelRunConfig, ModelRun, ModelLossHistory
        import Backend.app.models.latent_models    # registers LatentPoint and others
        # ... import any additional model modules here ...

        # Patch every ARRAY column to JSON for SQLite
        for table in _db.metadata.sorted_tables:
            for column in table.columns:
                if isinstance(column.type, sqltypes.ARRAY):
                    column.type = _db.JSON()

        # Create all tables with patched types
        _db.create_all()

    yield app

    # Teardown: remove session and drop all tables
    with app.app_context():
        _db.session.remove()
        _db.drop_all()


@pytest.fixture(scope="session")
def client(app):
    """
    Return a Flask test client.
    """
    return app.test_client()


@pytest.fixture(scope="function", autouse=True)
def session(app):
    """
    Provide a fresh SQLAlchemy session for each test function,
    using a SAVEPOINT so that tests are isolated.
    """
    # Get the engine directly for this app (no app context needed)
    engine = _db.get_engine(app)
    connection = engine.connect()
    transaction = connection.begin()

    # Bind a scoped session to the same connection
    sess = scoped_session(sessionmaker(bind=connection))
    _db.session = sess

    yield sess

    # Roll back the transaction and remove the session
    transaction.rollback()
    sess.remove()
    connection.close()
