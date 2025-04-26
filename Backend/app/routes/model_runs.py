# Backend/app/routes/model_runs.py

from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.run_models import ModelRun, ModelLossHistory

model_runs_bp = Blueprint(
    'model_runs',
    __name__,
    url_prefix='/model-runs'
)

@model_runs_bp.route('', methods=['GET'])
def list_model_runs():
    """
    (Optional) List all model runs.
    Currently returns an empty array, can be extended later.
    """
    return jsonify([])

@model_runs_bp.route('/<int:run_id>/losses', methods=['GET'])
def get_loss_history(run_id):
    """
    Return all loss and KL divergence entries for a given model run.
    """
    run = db.session.query(ModelRun).get(run_id)
    if not run:
        return jsonify([])

    history = (
        db.session.query(ModelLossHistory)
        .filter_by(model_run_id=run.id)
        .order_by(ModelLossHistory.epoch.asc())
        .all()
    )

    return jsonify([h.to_dict() for h in history])
