from flask import Blueprint, jsonify, request, current_app, abort
from threading import Thread
import requests

import logging

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig, ModelRun
from Pytorch.trainers.vae_trainer import train_vae_for_config, generate_and_bulk_save_latents

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
    Fetch a single ModelRunConfig by ID.
    Query param `relations=true` includes run, pca_configs, som_configs.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    include = request.args.get('relations', 'false').lower() == 'true'
    return jsonify(cfg.to_dict(include_relations=include)), 200

@model_configs_bp.route('', methods=['POST'])
def create_model_config():
    """
    Create a new ModelRunConfig.
    Expects JSON with keys: model_type (str), parameters (dict), currency_pairs (list of str)
    """
    data = request.get_json() or {}
    for field in ('model_type', 'parameters', 'currency_pairs'):
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
    Delete a ModelRunConfig and all associated run and loss history.
    """
    cfg = ModelRunConfig.query.get_or_404(config_id)
    db.session.delete(cfg)
    db.session.commit()
    return '', 204

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/model_configs.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@model_configs_bp.route('/<int:config_id>/train', methods=['POST'])
def trigger_training(config_id):
    """
    Trigger VAE training + latent generation for the given ModelRunConfig (sync).
    """
    logger.info(f"Starting training for config {config_id}")
    try:
        # 1) Validate config
        logger.info(f"Loading ModelRunConfig {config_id}...")
        cfg = ModelRunConfig.query.get_or_404(config_id)
        app = current_app._get_current_object()

        # 2a) Train the model
        logger.info("Calling train_vae_for_config...")
        run_id = train_vae_for_config(app, config_id)
        logger.info(f"Training completed, run_id: {run_id}")

        # 2b) Generate & bulk-save latents
        logger.info("Calling generate_and_bulk_save_latents...")
        generate_and_bulk_save_latents(app, run_id)
        logger.info(f"Latent generation completed for run {run_id}")

        logger.info(f"Training and latent generation completed for config {config_id}")
        return jsonify({"status": "completed", "run_id": run_id}), 200
    except Exception as e:
        logger.exception(f"Training or latent generation failed for config {config_id}: {str(e)}")
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# Other routes (e.g., create_config, get_losses) assumed unchanged

