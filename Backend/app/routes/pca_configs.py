from flask import Blueprint, request, jsonify, abort, current_app
from threading import Thread

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig
from Backend.app.models.latent_models import PCAProjectionConfig
from Pytorch.trainers.pca_trainer import run_pca_for_config

from sqlalchemy.orm import joinedload
import numpy as np
from Pytorch.projections.pca import fit_pca

# Blueprint for PCA configs under ModelRunConfig: /model-configs/<model_cfg_id>/pca-configs
pca_bp = Blueprint(
    'pca_configs',
    __name__,
    url_prefix='/model-configs'
)

@pca_bp.route('/<int:model_cfg_id>/pca-configs', methods=['GET'])
def list_pca_configs(model_cfg_id):
    """
    List all PCAProjectionConfig entries for a given model config.
    404 if the parent ModelRunConfig does not exist.
    """
    # validate parent config exists
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # fetch and serialize
    configs = PCAProjectionConfig.query.filter_by(model_config_id=model_cfg_id).all()
    return jsonify([c.to_dict() for c in configs]), 200

@pca_bp.route('/<int:model_cfg_id>/pca-configs/<int:pca_cfg_id>', methods=['GET'])
def get_pca_config(model_cfg_id, pca_cfg_id):
    """
    Retrieve a single PCAProjectionConfig by its ID under a given model config.
    404 if not found.
    """
    # validate parent config
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # fetch and serialize
    c = PCAProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=pca_cfg_id
    ).first_or_404(description="PCA config not found")
    return jsonify(c.to_dict()), 200

@pca_bp.route('/<int:model_cfg_id>/pca-configs', methods=['POST'])
def create_pca_config(model_cfg_id):
    """
    Create a new PCAProjectionConfig for a model config.
    Expects JSON with keys:
      - 'n_components' (int, required)
      - 'additional_params' (dict, optional)
    404 if parent config not found; 400 if missing required field.
    """
    # validate parent
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    data = request.get_json() or {}
    if 'n_components' not in data:
        abort(400, description="Missing required field: n_components")

    cfg = PCAProjectionConfig(
        model_config_id   = model_cfg_id,
        n_components      = data['n_components'],
        additional_params = data.get('additional_params', {})
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@pca_bp.route('/<int:model_cfg_id>/pca-configs/<int:pca_cfg_id>', methods=['PATCH', 'PUT'])
def update_pca_config(model_cfg_id, pca_cfg_id):
    """
    Update fields of an existing PCAProjectionConfig.
    Can modify 'n_components' and/or 'additional_params'.
    404 if parent or PCA config not found.
    """
    # validate parent
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # load PCA config
    cfg = PCAProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=pca_cfg_id
    ).first_or_404(description="PCA config not found")

    data = request.get_json() or {}
    if 'n_components' in data:
        cfg.n_components = data['n_components']
    if 'additional_params' in data:
        cfg.additional_params = data['additional_params']

    db.session.commit()
    return jsonify({'status': 'ok'}), 200

@pca_bp.route('/<int:model_cfg_id>/pca-configs/<int:pca_cfg_id>', methods=['DELETE'])
def delete_pca_config(model_cfg_id, pca_cfg_id):
    """
    Delete a PCAProjectionConfig and its related projections.
    404 if parent or PCA config not found.
    """
    # validate parent
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # load and delete
    cfg = PCAProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=pca_cfg_id
    ).first_or_404(description="PCA config not found")
    db.session.delete(cfg)
    db.session.commit()
    return '', 204

@pca_bp.route('/<int:model_cfg_id>/pca-configs/<int:pca_cfg_id>/train', methods=['POST'])
def trigger_pca_training(model_cfg_id, pca_cfg_id):
    """
    Trigger PCA training asynchronously for the existing PCAProjectionConfig.
    """
    # load the PCA config to get hyperparams
    cfg = PCAProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=pca_cfg_id
    ).first_or_404(description="PCA config not found")

    app = current_app._get_current_object()
    Thread(
        target=run_pca_for_config,
        args=(app, model_cfg_id, pca_cfg_id),
        daemon=True
    ).start()
    return jsonify({'status': 'PCA training started'}), 202