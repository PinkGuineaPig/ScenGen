from flask import Blueprint, jsonify, request, abort, make_response
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
    # collect all currency IDs
    currency_ids = {cid for b, q in rows for cid in (b, q)}
    # map IDs → codes
    codes = {
        c.id: c.code
        for c in db.session.query(Currency)
                           .filter(Currency.id.in_(currency_ids))
                           .all()
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
    Returns JSON array of {timestamp: ISO8601, rate: float}.
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

    # 2) Build base query
    query = ExchangeRate.query.filter_by(
        base_currency_id  = base.id,
        quote_currency_id = quote.id
    )

    # 3) Parse optional date filters
    start_str = request.args.get('start')
    end_str   = request.args.get('end')

    def parse_dt(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        abort(400, description=f"Invalid date format: {s}")

    start_dt = end_dt = None
    if start_str:
        start_dt = parse_dt(start_str)
    if end_str:
        end_dt = parse_dt(end_str)
    if start_dt and end_dt and start_dt > end_dt:
        abort(400, description="`start` must be before `end`")

    if start_dt:
        query = query.filter(ExchangeRate.timestamp >= start_dt)
    if end_dt:
        query = query.filter(ExchangeRate.timestamp <= end_dt)

    # 4) Execute
    rates = query.order_by(ExchangeRate.timestamp.asc()).all()

    # 5) Handle no‐content
    if not rates:
        return '', 204

    # 6) Serialize
    data = [
        {
            "timestamp": r.timestamp.isoformat(),
            "rate":       float(r.close)
        }
        for r in rates
    ]

    # 7) Return with caching header
    resp = make_response(jsonify(data), 200)
    resp.headers['Cache-Control'] = 'public, max-age=3600'
    return resp
