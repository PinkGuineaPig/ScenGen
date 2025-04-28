# Backend/app/routes/pca_projections.py

from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.latent_models import PCAProjection


from flask import Blueprint, jsonify, abort
from Backend.app.models.latent_models import PCAProjection, PCAProjectionConfig

pca_projections_bp = Blueprint(
    'pca_projections', __name__,
    url_prefix='/pca-projections/<int:pca_cfg_id>'
)

@pca_projections_bp.route('', methods=['GET'])
def list_pca_projections(pca_cfg_id):
    """
    Returns all PCAProjection rows for the given PCAProjectionConfig.
    GET /api/pca-projections/<pca_cfg_id>
    """
    # fetch all projections for that config
    projections = PCAProjection.query.filter_by(config_id=pca_cfg_id).all()

    # serialize
    result = [
        {
            'latent_point_id': proj.latent_point_id,
            'dim':              proj.dim,
            'value':            proj.value,
            'created_at':       proj.created_at.isoformat()
        }
        for proj in projections
    ]
    return jsonify(result), 200


@pca_projections_bp.route('/explained_variance', methods=['GET'])
def get_pca_explained_variance(pca_cfg_id):
    """
    Returns the stored explained‐variance ratio for the given PCAProjectionConfig.
    GET /api/pca-projections/<pca_cfg_id>/explained_variance
    """
    # load the PCA config (404 if not found)
    cfg = PCAProjectionConfig.query.get_or_404(
        pca_cfg_id,
        description="PCA config not found"
    )

    if cfg.explained_variance is None:
        abort(404, description="Explained variance not available (run PCA first)")

    return jsonify(cfg.explained_variance), 200