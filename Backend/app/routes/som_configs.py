from flask import Blueprint, request, jsonify, abort, current_app
from threading import Thread

from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig
from Backend.app.models.latent_models import SOMProjectionConfig
from Pytorch.trainers.som_trainer import train_som_for_config

# Blueprint for SOM configs under ModelRunConfig: /model-configs/<model_cfg_id>/som-configs
som_bp = Blueprint(
    'som_configs',
    __name__,
    url_prefix='/model-configs'
)

@som_bp.route('/<int:model_cfg_id>/som-configs', methods=['GET'])
def list_som_configs(model_cfg_id):
    """
    List all SOMProjectionConfig entries for a given model config.
    404 if the parent ModelRunConfig does not exist.
    """
    # validate parent config exists
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # fetch and serialize
    configs = SOMProjectionConfig.query.filter_by(model_config_id=model_cfg_id).all()
    return jsonify([c.to_dict() for c in configs]), 200

@som_bp.route('/<int:model_cfg_id>/som-configs/<int:som_cfg_id>', methods=['GET'])
def get_som_config(model_cfg_id, som_cfg_id):
    """
    Retrieve a single SOMProjectionConfig by its ID under a given model config.
    404 if not found.
    """
    # validate parent
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")

    # fetch and serialize
    c = SOMProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=som_cfg_id
    ).first_or_404(description="SOM config not found")
    return jsonify(c.to_dict()), 200

@som_bp.route('/<int:model_cfg_id>/som-configs', methods=['POST'])
def create_som_config(model_cfg_id):
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")
    data = request.get_json() or {}
    for field in ('x_dim', 'y_dim', 'iterations'):
        if field not in data:
            abort(400, description=f"Missing required field: {field}")
    cfg = SOMProjectionConfig(
        model_config_id   = model_cfg_id,
        x_dim             = data['x_dim'],
        y_dim             = data['y_dim'],
        iterations        = data['iterations'],
        additional_params = data.get('additional_params', {})
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@som_bp.route('/<int:model_cfg_id>/som-configs/<int:som_cfg_id>', methods=['PATCH', 'PUT'])
def update_som_config(model_cfg_id, som_cfg_id):
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")
    cfg = SOMProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=som_cfg_id
    ).first_or_404(description="SOM config not found")
    data = request.get_json() or {}
    if 'x_dim' in data:
        cfg.x_dim = data['x_dim']
    if 'y_dim' in data:
        cfg.y_dim = data['y_dim']
    if 'iterations' in data:
        cfg.iterations = data['iterations']
    if 'additional_params' in data:
        cfg.additional_params = data['additional_params']
    db.session.commit()
    return jsonify({'status': 'ok'}), 200

@som_bp.route('/<int:model_cfg_id>/som-configs/<int:som_cfg_id>', methods=['DELETE'])
def delete_som_config(model_cfg_id, som_cfg_id):
    ModelRunConfig.query.get_or_404(model_cfg_id, description="Model config not found")
    cfg = SOMProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=som_cfg_id
    ).first_or_404(description="SOM config not found")
    db.session.delete(cfg)
    db.session.commit()
    return '', 204

@som_bp.route('/<int:model_cfg_id>/som-configs/<int:som_cfg_id>/train', methods=['POST'])
def trigger_som_training(model_cfg_id, som_cfg_id):
    # validate config exists
    SOMProjectionConfig.query.filter_by(
        model_config_id=model_cfg_id,
        id=som_cfg_id
    ).first_or_404(description="SOM config not found")
    app = current_app._get_current_object()
    Thread(
        target=_train_som_worker,
        args=(app, model_cfg_id, som_cfg_id),
        daemon=True
    ).start()
    return jsonify({'status': 'SOM training started'}), 202


def _train_som_worker(app, model_cfg_id, som_cfg_id):
    """
    Background worker to invoke SOM trainer.
    """
    train_som_for_config(app, model_cfg_id, som_cfg_id)
