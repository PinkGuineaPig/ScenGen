# Backend/app/api/pca_configs.py

from flask import Blueprint, request, jsonify, abort
from Backend.app import db
from Backend.app.models.run_models import ModelRun
from Backend.app.models.latent_models import PCAProjectionConfig

pca_bp = Blueprint(
    'pca_configs',
    __name__,
    url_prefix='/runs/<int:run_id>/pca-configs'
)

@pca_bp.route('', methods=['GET'])
def list_configs(run_id):
    """
    List all PCAProjectionConfig entries for a given model run.
    404 if the run does not exist.
    """
    # 1) ensure run exists
    run = ModelRun.query.filter_by(id=run_id) \
                       .first_or_404(description="Run not found")

    # 2) pull all PCA configs for that run's config_id
    configs = (PCAProjectionConfig.query
               .filter_by(config_id=run.config_id)
               .all())

    # 3) serialize
    result = [
        {
            'id':                c.id,
            'n_components':      c.n_components,
            'additional_params': c.additional_params
        }
        for c in configs
    ]
    return jsonify(result), 200


@pca_bp.route('/<int:cfg_id>', methods=['GET'])
def get_config(run_id, cfg_id):
    """
    Retrieve a single PCAProjectionConfig by id within a model run.
    404 if run or config not found.
    """
    # ensure run exists
    run = ModelRun.query.filter_by(id=run_id) \
                       .first_or_404(description="Run not found")

    # fetch config by matching its config_id to run.config_id
    cfg = (PCAProjectionConfig.query
           .filter_by(config_id=run.config_id, id=cfg_id)
           .first_or_404(description="Config not found"))

    return jsonify({
        'id':                cfg.id,
        'n_components':      cfg.n_components,
        'additional_params': cfg.additional_params
    }), 200


@pca_bp.route('', methods=['POST'])
def create_config(run_id):
    """
    Create a new PCAProjectionConfig for a model run.
    Expects JSON with:
      - 'n_components' (int, required)
      - 'additional_params' (dict, optional)
    404 if run not found; 400 if missing required field.
    """
    # ensure run exists
    run = ModelRun.query.filter_by(id=run_id) \
                       .first_or_404(description="Run not found")

    data = request.get_json() or {}
    if 'n_components' not in data:
        abort(400, description="Missing required field: n_components")

    cfg = PCAProjectionConfig(
        model_run_id      = run.id,              # ← satisfy NOT NULL
        config_id         = run.config_id,
        n_components      = data['n_components'],
        additional_params = data.get('additional_params', {})
    )
    
    db.session.add(cfg)
    db.session.commit()

    return jsonify({'id': cfg.id}), 201


@pca_bp.route('/<int:cfg_id>', methods=['PATCH', 'PUT'])
def update_config(run_id, cfg_id):
    """
    Update an existing PCAProjectionConfig.
    Can modify 'n_components' and/or 'additional_params'.
    404 if run or config not found.
    """
    # ensure run exists
    run = ModelRun.query.filter_by(id=run_id) \
                       .first_or_404(description="Run not found")

    # load the specific PCA config under that run
    cfg = (PCAProjectionConfig.query
           .filter_by(config_id=run.config_id, id=cfg_id)
           .first_or_404(description="Config not found"))

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
    Delete a PCAProjectionConfig.
    404 if run or config not found.
    """
    # ensure run exists
    run = ModelRun.query.filter_by(id=run_id) \
                       .first_or_404(description="Run not found")

    # load and delete
    cfg = (PCAProjectionConfig.query
           .filter_by(config_id=run.config_id, id=cfg_id)
           .first_or_404(description="Config not found"))

    db.session.delete(cfg)
    db.session.commit()
    return '', 204
