# Backend/app/models/data_models.py
# ---------------------------
# DataGroup: Used to map all available data across different tables.
#            Exposes filterable options to the frontend.


from Backend.app import db
from datetime import datetime

class DataGroup(db.Model):
    __tablename__ = 'data_group'

    id = db.Column(db.Integer, primary_key=True)
    table_name = db.Column(db.String(80), nullable=False)
    subgroup = db.Column(db.String(300), nullable=False)

    real_estate_entries = db.relationship('RealEstateData', back_populates='data_group')
    labour_market_entries = db.relationship('LabourMarketData', back_populates='data_group')
    bond_yield_entries = db.relationship('BondYieldsData', back_populates='data_group')

    def __repr__(self):
        # Provides a readable representation for debugging/logging
        return f'<DataGroup {self.table_name} | {self.subgroup}>'

    @classmethod
    def get_all(cls, session):
        """
        Fetches all DataGroup records.
        Used by the frontend to list available data categories.
        """
        return session.query(cls).all()

# --------------------------------------------------------------------
# RealEstateData: Stores real estate metrics for selected regions and types.
class RealEstateData(db.Model):
    __tablename__ = 'real_estate_data'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    value = db.Column(db.Float, nullable=False)
    property_type = db.Column(db.String(80), nullable=False)
    region = db.Column(db.String(80), nullable=False)
    price_type = db.Column(db.String(80), nullable=False)

    data_group_id = db.Column(db.Integer, db.ForeignKey('data_group.id'), nullable=False)
    data_group = db.relationship('DataGroup', back_populates='real_estate_entries')

    def __repr__(self):
        # Human-readable for logs
        return f'<RealEstateData {self.date} {self.value}>'

    @classmethod
    def get_all(cls, session):
        """
        Retrieves every real estate data entry.
        Used to populate time-series plots and analyses.
        """
        return session.query(cls).all()

    @classmethod
    def get_by_group_ids(cls, session, group_ids):
        """
        Filters real estate data by one or more DataGroup IDs.
        Called when users select specific subgroups on the UI.
        """
        return session.query(cls).filter(cls.data_group_id.in_(group_ids)).all()

# --------------------------------------------------------------------
# LabourMarketData: Stores labor market statistics (e.g., unemployment rate).
class LabourMarketData(db.Model):
    __tablename__ = 'labour_market_data'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    value = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(80), nullable=False)

    data_group_id = db.Column(db.Integer, db.ForeignKey('data_group.id'), nullable=False)
    data_group = db.relationship('DataGroup', back_populates='labour_market_entries')

    def __repr__(self):
        return f'<LabourMarketData {self.date} {self.value}>'

    @classmethod
    def get_all(cls, session):
        """
        Fetches all labour market entries.
        Used for dashboard statistics if no filters applied.
        """
        return session.query(cls).all()

    @classmethod
    def get_by_type(cls, session, data_type):
        """
        Filters labour data by type (e.g., "unemployment").
        Triggered when frontend selects a specific labour metric.
        """
        return session.query(cls).filter_by(type=data_type).all()

    @classmethod
    def get_unique_types(cls, session):
        """
        Returns distinct types available (for filter dropdowns).
        Populates the labour metrics selector in the UI.
        """
        return [t[0] for t in session.query(cls.type).distinct().all()]

    @classmethod
    def get_by_group_ids(cls, session, group_ids):
        """
        Filters labour market data by DataGroup IDs.
        Used when multiple subgroups are selected.
        """
        return session.query(cls).filter(cls.data_group_id.in_(group_ids)).all()

# --------------------------------------------------------------------
# BondYieldsData: Represents bond yield curves or benchmarks.
class BondYieldsData(db.Model):
    __tablename__ = 'bond_yields_data'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    value = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(300), nullable=False)

    data_group_id = db.Column(db.Integer, db.ForeignKey('data_group.id'), nullable=False)
    data_group = db.relationship('DataGroup', back_populates='bond_yield_entries')

    def __repr__(self):
        return f'<BondYieldsData {self.date} {self.value}>'

    @classmethod
    def get_all(cls, session):
        """
        Fetches all bond yield records.
        Used for time-series comparison and overlays.
        """
        return session.query(cls).all()

    @classmethod
    def get_by_type(cls, session, data_type):
        """
        Filters yields by type (e.g., "10-year").
        Populates type filter in the bond yields panel.
        """
        return session.query(cls).filter_by(type=data_type).all()

    @classmethod
    def get_unique_types(cls, session):
        """
        Lists available yield types for UI selectors.
        """
        return [t[0] for t in session.query(cls.type).distinct().all()]

    @classmethod
    def get_by_group_ids(cls, session, group_ids):
        """
        Filters bond yields by DataGroup IDs.
        Supports multi-select in the frontend.
        """
        return session.query(cls).filter(cls.data_group_id.in_(group_ids)).all()
    





    # Currency: Lookup table for currency codes (e.g., EUR, USD).
#           Normalizes the set of currencies for FX data.
class Currency(db.Model):
    __tablename__ = 'currency'

    id   = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(3), unique=True, nullable=False)
    name = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<Currency {self.code}>'


# ExchangeRate: Time-series table storing FX rates between currency pairs.
#               Rate is quote_amount per 1 base_amount at a given timestamp.
class ExchangeRate(db.Model):
    __tablename__ = 'exchange_rate'

    id                = db.Column(db.Integer, primary_key=True)
    base_currency_id  = db.Column(db.Integer, db.ForeignKey('currency.id'), nullable=False)
    quote_currency_id = db.Column(db.Integer, db.ForeignKey('currency.id'), nullable=False)
    rate              = db.Column(db.Numeric(18,8), nullable=False)
    timestamp         = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint('base_currency_id', 'quote_currency_id', 'timestamp', name='uq_fx_pair_time'),
    )

    # Relationships back to the Currency lookup
    base_currency  = db.relationship('Currency', foreign_keys=[base_currency_id])
    quote_currency = db.relationship('Currency', foreign_keys=[quote_currency_id])

    def __repr__(self):
        bc = self.base_currency.code
        qc = self.quote_currency.code
        ts = self.timestamp.isoformat()
        return f'<ExchangeRate {bc}->{qc} @ {ts} = {self.rate}>'
