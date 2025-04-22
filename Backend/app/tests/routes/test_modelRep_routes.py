import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.app.models.run_models     import ModelRun, ModelRunConfig


# Use an in-memory SQLite database for fast, isolated tests
TEST_DB_URI = "sqlite:///:memory:"

@pytest.fixture(scope="module")
def engine():
    return create_engine(TEST_DB_URI)

@pytest.fixture(scope="module")
def tables(engine):
    # Create tables for Currency and ExchangeRate
    ModelRun.metadata.create_all(engine)
    ModelRunConfig.metadata.create_all(engine)
    yield
    # Drop tables after tests
    ModelRunConfig.metadata.drop_all(engine)
    ModelRun.metadata.drop_all(engine)

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



#    'id' = 23
#    'created_at' = datetime.today(),
#    'config_id' = db.Column(db.Integer, db.ForeignKey('model_run_config.id'))
#    'config' = db.relationship('ModelRunConfig')
#    'group_ids' = [1,2,3]
#    'version' = 1
#    'model_blob' = db.Column(db.LargeBinary)
