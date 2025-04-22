# Backend/app/routes/model_runs.py

from flask import Blueprint, jsonify

model_runs_bp = Blueprint(
    'model_runs',
    __name__,
    url_prefix='/api/model_runs'
)

@model_runs_bp.route('', methods=['GET'])
def list_model_runs():
    """Stub to list model runs (empty for now)."""
    return jsonify([])
