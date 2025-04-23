# Backend/app/routes/model_configs.py

from flask import Blueprint, jsonify, request, current_app, abort
from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig
from threading import Thread
from Pytorch.trainers.vae_trainer import train_vae_for_config

# Blueprint mounted under /api by app factory
model_configs_bp = Blueprint('model_configs', __name__, url_prefix='/model-configs')

@model_configs_bp.route('', methods=['GET'])
def list_model_configs():
    """
    List all ModelRunConfig entries.
    """
    configs = ModelRunConfig.query.all()
    return jsonify([cfg.to_dict() for cfg in configs]), 200

@model_configs_bp.route('/<int:config_id>', methods=['GET'])
def get_model_config(config_id):
    """
    Fetch a single ModelRunConfig by ID. Query param `relations=true` includes runs and loss history.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    include = request.args.get('relations', 'false').lower() == 'true'
    return jsonify(cfg.to_dict(include_relations=include)), 200

@model_configs_bp.route('', methods=['POST'])
def create_model_config():
    """
    Create a new ModelRunConfig. Expects JSON with keys:
      - model_type (str)
      - parameters (dict)
      - currency_pairs (list of str)
    """
    data = request.get_json() or {}
    required = ['model_type', 'parameters', 'currency_pairs']
    for field in required:
        if field not in data:
            abort(400, f"Missing required field: {field}")

    cfg = ModelRunConfig(
        model_type=data['model_type'],
        parameters=data['parameters'],
        currency_pairs=data['currency_pairs']
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@model_configs_bp.route('/<int:config_id>', methods=['PATCH', 'PUT'])
def update_model_config(config_id):
    """
    Update fields of an existing ModelRunConfig.
    Accepts any of: model_type, parameters, currency_pairs.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    data = request.get_json() or {}
    if 'model_type' in data:
        cfg.model_type = data['model_type']
    if 'parameters' in data:
        cfg.parameters = data['parameters']
    if 'currency_pairs' in data:
        cfg.currency_pairs = data['currency_pairs']
    db.session.commit()
    return jsonify({'status': 'ok'}), 200

@model_configs_bp.route('/<int:config_id>', methods=['DELETE'])
def delete_model_config(config_id):
    """
    Delete a ModelRunConfig and all associated runs and loss history.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    db.session.delete(cfg)
    db.session.commit()
    return '', 204

@model_configs_bp.route('/<int:config_id>/train', methods=['POST'])
def trigger_training(config_id):
    """
    Trigger VAE training for the given ModelRunConfig (async).
    Uses the stored currency_pairs and parameters on the config.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    app = current_app._get_current_object()
    # Start training in background thread
    Thread(
        target=train_vae_for_config,
        args=(app, cfg.currency_pairs, cfg.parameters)
    ).start()
    return jsonify({'status': 'training started'}), 202


