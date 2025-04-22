from flask import Blueprint, request, jsonify, abort
from Backend.app import db
from Backend.app.models.latent_models import PCAProjectionConfig
from Backend.app.models.run_models import ModelRun

# Blueprint for PCA config CRUD under /api/runs/<run_id>/pca-configs
pca_bp = Blueprint('pca_configs', __name__, url_prefix='/runs/<int:run_id>/pca-configs')

@pca_bp.route('', methods=['GET'])
def list_configs(run_id):
    """
    List all PCAProjectionConfig entries for a given model run.
    404 if the run does not exist.
    """
    # Ensure the model run exists or 404
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")

    configs = PCAProjectionConfig.query.filter_by(model_run_id=run_id).all()
    result = [
        {
            'id':                c.id,
            'n_components':      c.n_components,
            'additional_params': c.additional_params
        } for c in configs
    ]
    return jsonify(result), 200

@pca_bp.route('/<int:cfg_id>', methods=['GET'])
def get_config(run_id, cfg_id):
    """
    Retrieve a single PCAProjectionConfig by id within a model run.
    404 if run or config not found.
    """
    # Ensure run exists
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    # Fetch config or 404
    cfg = PCAProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404(description="Config not found")

    return jsonify({
        'id':                cfg.id,
        'n_components':      cfg.n_components,
        'additional_params': cfg.additional_params
    }), 200

@pca_bp.route('', methods=['POST'])
def create_config(run_id):
    """
    Create a new PCAProjectionConfig for a model run.
    Expects JSON with 'n_components' (int), optional 'additional_params' (dict).
    404 if run not found; 400 if missing required field.
    """
    # Ensure run exists
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")

    data = request.get_json() or {}
    if 'n_components' not in data:
        abort(400, description="Missing required field: n_components")

    cfg = PCAProjectionConfig(
        model_run_id=run_id,
        n_components=data['n_components'],
        additional_params=data.get('additional_params', {})
    )
    db.session.add(cfg)
    db.session.commit()
    return jsonify({'id': cfg.id}), 201

@pca_bp.route('/<int:cfg_id>', methods=['PATCH', 'PUT'])
def update_config(run_id, cfg_id):
    """
    Update an existing PCAProjectionConfig. Can modify 'n_components' and/or 'additional_params'.
    404 if run or config not found.
    """
    # Ensure run and config exist
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    cfg = PCAProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404(description="Config not found")

    data = request.get_json() or {}
    if 'n_components' in data:
        cfg.n_components = data['n_components']
    if 'additional_params' in data:
        cfg.additional_params = data['additional_params']

    db.session.commit()
    return jsonify({'status': 'ok'}), 200

@pca_bp.route('/<int:cfg_id>', methods=['DELETE'])
def delete_config(run_id, cfg_id):
    """
    Delete a PCAProjectionConfig and its projections.
    404 if run or config not found.
    """
    # Ensure run and config exist
    ModelRun.query.filter_by(id=run_id).first_or_404(description="Run not found")
    cfg = PCAProjectionConfig.query.filter_by(model_run_id=run_id, id=cfg_id).first_or_404(description="Config not found")

    db.session.delete(cfg)
    db.session.commit()
    # Return no content per HTTP 204 conventions
    return '', 204
