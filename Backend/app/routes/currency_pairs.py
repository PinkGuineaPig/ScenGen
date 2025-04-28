from flask import Blueprint, jsonify, request, abort
from Backend.app import db
from Backend.app.models.data_models import ExchangeRate, Currency
from datetime import datetime

currency_bp = Blueprint('currency_pairs', __name__, url_prefix='/currency-pairs')

@currency_bp.route('', methods=['GET'])
def list_currency_pairs():
    rows = (
        db.session.query(
            ExchangeRate.base_currency_id,
            ExchangeRate.quote_currency_id
        )
        .distinct()
        .all()
    )
    currency_ids = {cid for b, q in rows for cid in (b, q)}
    codes = {
        c.id: c.code
        for c in db.session.query(Currency).filter(Currency.id.in_(currency_ids)).all()
    }
    pairs = sorted(f"{codes[b]}/{codes[q]}" for b, q in rows)
    return jsonify(pairs), 200

@currency_bp.route('/<path:pair>/history', methods=['GET'])
def get_currency_history(pair):
    """
    GET /currency-pairs/<BASE>/<QUOTE>/history
    Optional query params:
      - start=YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
      - end=  YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
    Returns JSON array of {date: ISO8601, rate: float}.
    """
    # 1) Parse and validate pair
    try:
        base_code, quote_code = pair.split('/')
    except ValueError:
        abort(400, description="Pair must be in BASE/QUOTE format")
    base = Currency.query.filter_by(code=base_code.upper()).one_or_none()
    quote = Currency.query.filter_by(code=quote_code.upper()).one_or_none()
    if not base or not quote:
        abort(404, description="Unknown currency code")

    # 2) Build query for the requested time series
    query = ExchangeRate.query.filter_by(
        base_currency_id  = base.id,
        quote_currency_id = quote.id
    )

    # 3) Apply optional date filters
    start_str = request.args.get('start')
    end_str   = request.args.get('end')
    def parse_dt(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        abort(400, description=f"Invalid date format: {s}")

    if start_str:
        start_dt = parse_dt(start_str)
        query = query.filter(ExchangeRate.timestamp >= start_dt)
    if end_str:
        end_dt = parse_dt(end_str)
        query = query.filter(ExchangeRate.timestamp <= end_dt)

    # 4) Execute and serialize
    rates = query.order_by(ExchangeRate.timestamp.asc()).all()
    data = [
        {
            "date":  r.timestamp.isoformat(),
            "rate":  float(r.close)
        }
        for r in rates
    ]

    return jsonify(data), 200
