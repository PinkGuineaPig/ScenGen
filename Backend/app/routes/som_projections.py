# Backend/app/routes/som_projections.py

from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.run_models import ModelRun
from Backend.app.models.latent_models import SOMProjection
from sqlalchemy.orm import joinedload

som_projections_bp = Blueprint(
    'som_projections',
    __name__,
    url_prefix='/som-projections/<int:som_cfg_id>'
)


@som_projections_bp.route('', methods=['GET'])
def list_som_projections(som_cfg_id):
    """
    Returns all SOMProjection rows for the given som_cfg_id,
    including the corresponding LatentPoint.start_date.
    """
    # 1) fetch all projections for that config, eager‐loading the related LatentPoint
    projections = (
        SOMProjection.query
        .options(joinedload(SOMProjection.point))
        .filter_by(config_id=som_cfg_id)
        .all()
    )

    # 2) serialize, now including start_date
    result = [
        {
            'latent_point_id': proj.latent_point_id,
            'start_date':      proj.point.start_date.isoformat(),
            'x':               proj.x,
            'y':               proj.y,
            'created_at':      proj.created_at.isoformat()
        }
        for proj in projections
    ]

    return jsonify(result), 200