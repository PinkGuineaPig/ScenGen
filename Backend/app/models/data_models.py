# Backend/app/models/data_models.py
# ---------------------------

from Backend.app import db

    # Currency: Lookup table for currency codes (e.g., EUR, USD).
    # Normalizes the set of currencies for FX data.
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
    open              = db.Column(db.Numeric(18,8), nullable=False)
    high              = db.Column(db.Numeric(18,8), nullable=False)
    low               = db.Column(db.Numeric(18,8), nullable=False)
    close             = db.Column(db.Numeric(18,8), nullable=False)
    timestamp         = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            'base_currency_id',
            'quote_currency_id',
            'timestamp',
            name='uq_fx_pair_time'
        ),
    )

    base_currency  = db.relationship('Currency', foreign_keys=[base_currency_id])
    quote_currency = db.relationship('Currency', foreign_keys=[quote_currency_id])

    def __repr__(self):
        return (
            f'<ExchangeRate {self.base_currency.code}->{self.quote_currency.code} '
            f'@ {self.timestamp.date()} O={self.open} H={self.high} '
            f'L={self.low} C={self.close}>'
        )