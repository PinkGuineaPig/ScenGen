# Backend/app/routes/som_projections.py

from flask import Blueprint, jsonify
from Backend.app import db
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
    including the full datetime of the associated LatentPoint.
    """
    # 1) fetch all projections for that config, eager‐loading the LatentPoint
    projections = (
        SOMProjection.query
        .options(joinedload(SOMProjection.point))
        .filter_by(config_id=som_cfg_id)
        .all()
    )

    # 2) serialize, including the full ISO timestamp
    result = []
    for proj in projections:
        # If you've renamed the column to start_datetime, adjust here accordingly
        dt = proj.point.start_date  # must be a datetime, not a date
        result.append({
            'latent_point_id': proj.latent_point_id,
            'start_timestamp': dt.isoformat(),    # full YYYY-MM-DDTHH:MM:SS
            'x':                proj.x,
            'y':                proj.y,
            'created_at':       proj.created_at.isoformat()
        })

    return jsonify(result), 200
