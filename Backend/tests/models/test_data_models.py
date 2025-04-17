import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from Backend.app import db, create_app
from Backend.app.models.data_models import DataGroup, RealEstateData

# Use an in-memory SQLite database for fast unit tests
TEST_DATABASE_URI = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    return create_engine(TEST_DATABASE_URI)

@pytest.fixture(scope="session")
def tables(engine):
    # Create all tables once per session
    db.metadata.create_all(engine)
    yield
    db.metadata.drop_all(engine)

@pytest.fixture()
def session(engine, tables):
    """
    Provides a transactional scope around a series of operations.
    Rolls back after each test for isolation.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


def test_datagroup_get_all_empty(session):
    """
    Edge case: No DataGroup records should return empty list.
    """
    result = DataGroup.get_all(session)
    assert result == []


def test_datagroup_and_realestate_relationship(session):
    """
    Basic case: Insert DataGroup and related RealEstateData,
    then verify queries return expected results.
    """
    # Insert a DataGroup
    dg = DataGroup(table_name="real_estate_data", subgroup="houses")
    session.add(dg)
    session.commit()

    # Verify get_all returns the new group
    groups = DataGroup.get_all(session)
    assert len(groups) == 1
    assert groups[0].table_name == "real_estate_data"

    # Add a RealEstateData linked to this group
    re = RealEstateData(
        date="2020-01-01", value=123.45,
        property_type="house", region="RegionA", price_type="median",
        data_group_id=dg.id
    )
    session.add(re)
    session.commit()

    # Query RealEstateData via get_all and by_group_ids
    all_re = RealEstateData.get_all(session)
    assert len(all_re) == 1
    by_group = RealEstateData.get_by_group_ids(session, [dg.id])
    assert len(by_group) == 1
    assert by_group[0].value == 123.45


def test_realestate_get_by_nonexistent_group(session):
    """
    Edge case: Filtering by a non-existent group ID returns empty list.
    """
    result = RealEstateData.get_by_group_ids(session, [999])
    assert result == []


def test_realestate_get_by_empty_group_list(session):
    """
    Edge case: Empty group_ids list should return empty list immediately.
    """
    result = RealEstateData.get_by_group_ids(session, [])
    assert result == []


def test_realestate_multiple_groups_filter(session):
    """
    Complex case: Insert two DataGroups and multiple RealEstateData entries,
    then filter by both IDs and verify both entries are returned.
    """
    # Create two groups
    dg1 = DataGroup(table_name="real_estate_data", subgroup="houses")
    dg2 = DataGroup(table_name="real_estate_data", subgroup="apartments")
    session.add_all([dg1, dg2])
    session.commit()

    # Add entries for each group
    re1 = RealEstateData(
        date="2020-01-01", value=100.0,
        property_type="house", region="A", price_type="median",
        data_group_id=dg1.id
    )
    re2 = RealEstateData(
        date="2020-02-01", value=200.0,
        property_type="apt", region="B", price_type="mean",
        data_group_id=dg2.id
    )
    session.add_all([re1, re2])
    session.commit()

    # Filter by both group IDs
    result = RealEstateData.get_by_group_ids(session, [dg1.id, dg2.id])
    assert len(result) == 2
    values = sorted([r.value for r in result])
    assert values == [100.0, 200.0]
