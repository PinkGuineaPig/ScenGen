from flask import Blueprint, request, jsonify, abort
from Backend.app import db
from Backend.app.models.latent_models import SOMProjectionConfig
from Backend.app.models.run_models import ModelRun  # Import your Run model

som_bp = Blueprint('som_configs', __name__, url_prefix='/runs/<int:run_id>/som-configs')

@som_bp.route('', methods=['GET'])
def list_configs(run_id):
    # 404 if there is no such run
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")

    configs = SOMProjectionConfig.query.filter_by(model_run_id=run_id).all()
    payload = [{
        'id':                c.id,
        'x_dim':             c.x_dim,
        'y_dim':             c.y_dim,
        'iterations':        c.iterations,
        'additional_params': c.additional_params,
    } for c in configs]

    return jsonify(payload), 200

@som_bp.route('/<int:cfg_id>', methods=['GET'])
def get_config(run_id, cfg_id):
    c = SOMProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404()
    return jsonify({
        'id':            c.id,
        'x_dim':         c.x_dim,
        'y_dim':         c.y_dim,
        'iterations':    c.iterations,
        'additional_params': c.additional_params,
    }), 200

@som_bp.route('', methods=['POST'])
def create_config(run_id):
    data = request.get_json()
    for field in ('x_dim','y_dim','iterations'):
        if field not in data:
            abort(400, f"Missing required field: {field}")
    cfg = SOMProjectionConfig(
        model_run_id=run_id,
        x_dim=data['x_dim'],
        y_dim=data['y_dim'],
        iterations=data['iterations'],
        additional_params=data.get('additional_params')
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@som_bp.route('/<int:cfg_id>', methods=['PUT','PATCH'])
def update_config(run_id, cfg_id):
    c = SOMProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404()
    data = request.get_json()
    if 'x_dim' in data:      c.x_dim = data['x_dim']
    if 'y_dim' in data:      c.y_dim = data['y_dim']
    if 'iterations' in data: c.iterations = data['iterations']
    if 'additional_params' in data:
        c.additional_params = data['additional_params']
    db.session.commit()
    return jsonify({'status':'ok'}), 200

@som_bp.route('/<int:cfg_id>', methods=['DELETE'])
def delete_config(run_id, cfg_id):
    c = SOMProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404()
    db.session.delete(c)
    db.session.commit()
    return jsonify({'status':'deleted'}), 204
