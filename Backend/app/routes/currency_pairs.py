# Backend/app/routes/currency_pairs.py
from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.data_models import ExchangeRate, Currency

currency_bp = Blueprint('currency_pairs', __name__, url_prefix='/currency-pairs')

@currency_bp.route('', methods=['GET'])
def list_currency_pairs():
    """
    Return all distinct available currency pairs in the form 'BASE/QUOTE'.
    """
    # Query distinct base and quote currency IDs
    rows = (
        db.session.query(
            ExchangeRate.base_currency_id,
            ExchangeRate.quote_currency_id
        )
        .distinct()
        .all()
    )

    # Fetch currency codes in one go
    currency_ids = set()
    for base_id, quote_id in rows:
        currency_ids.add(base_id)
        currency_ids.add(quote_id)
    currencies = (
        db.session.query(Currency)
        .filter(Currency.id.in_(currency_ids))
        .all()
    )
    id_to_code = {c.id: c.code for c in currencies}

    # Build pairs
    pairs = [f"{id_to_code[base]}/{id_to_code[quote]}" for base, quote in rows]
    # Sort and return
    pairs = sorted(pairs)

    return jsonify(pairs), 200
