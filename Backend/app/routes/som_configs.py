import json
from flask import Blueprint, request, jsonify, abort, current_app
from threading import Thread
from Backend.app import db
from Backend.app.models.latent_models import SOMProjectionConfig, SOMProjection
from Backend.app.models.run_models import ModelRun
from Pytorch.trainers.som_trainer import train_som_for_config

# Mounted under /api by create_app
som_bp = Blueprint(
    'som_configs', __name__, url_prefix='/runs/<int:run_id>/som-configs'
)

@som_bp.route('', methods=['GET'])
def list_configs(run_id):
    run = ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    configs = SOMProjectionConfig.query.filter_by(config_id=run.config_id).all()
    payload = [{
        'id':                c.id,
        'x_dim':             c.x_dim,
        'y_dim':             c.y_dim,
        'iterations':        c.iterations,
        'additional_params': c.additional_params,
    } for c in configs]
    return jsonify(payload), 200

@som_bp.route('', methods=['POST'])
def create_config(run_id):
    run = ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    data = request.get_json() or {}
    for field in ('x_dim','y_dim','iterations'):
        if field not in data:
            abort(400, description=f"Missing required field: {field}")
    cfg = SOMProjectionConfig(
        model_run_id      = run.id,
        config_id         = run.config_id,
        x_dim             = data['x_dim'],
        y_dim             = data['y_dim'],
        iterations        = data['iterations'],
        additional_params = data.get('additional_params', {})
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@som_bp.route('/<int:cfg_id>', methods=['PATCH', 'PUT'])
def update_config(run_id, cfg_id):
    run = ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    c = SOMProjectionConfig.query.filter_by(config_id=run.config_id, id=cfg_id).first_or_404(description="Config not found")

    data = request.get_json() or {}
    if 'x_dim' in data:
        c.x_dim = data['x_dim']
    if 'y_dim' in data:
        c.y_dim = data['y_dim']
    if 'iterations' in data:
        c.iterations = data['iterations']
    if 'additional_params' in data:
        c.additional_params = data['additional_params']

    db.session.commit()
    return jsonify({'status': 'ok'}), 200

@som_bp.route('/<int:cfg_id>', methods=['DELETE'])
def delete_config(run_id, cfg_id):
    run = ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    c = SOMProjectionConfig.query.filter_by(config_id=run.config_id, id=cfg_id).first_or_404(description="Config not found")
    db.session.delete(c)
    db.session.commit()
    return '', 204

@som_bp.route('/<int:cfg_id>/train', methods=['POST'])
def trigger_som_training(run_id, cfg_id):
    """
    Trigger SOM training asynchronously for the existing SOMProjectionConfig.
    """
    run = ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    cfg = SOMProjectionConfig.query.filter_by(config_id=run.config_id, id=cfg_id).first_or_404(description="Config not found")

    app = current_app._get_current_object()
    # Start background training
    Thread(
        target=train_som_for_config,
        args=(app, run_id, cfg_id),
        daemon=True
    ).start()

    return jsonify({'status': 'SOM training started'}), 202
