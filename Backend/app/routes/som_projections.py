# Backend/app/routes/som_projections.py

from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.run_models import ModelRun
from Backend.app.models.latent_models import SOMProjection

som_projections_bp = Blueprint(
    'som_projections',
    __name__,
    url_prefix='/runs/<int:run_id>/som-projections'
)

@som_projections_bp.route('', methods=['GET'])
def list_som_projections(run_id):
    """
    Returns all SOM projections for a given run.
    """
    run = db.session.query(ModelRun).get(run_id)
    if not run:
        return jsonify([]), 404

    projections = (
        db.session.query(SOMProjection)
        .join(SOMProjection.point)
        .filter(SOMProjection.config_id.in_(
            [cfg.id for cfg in run.config.som_configs]
        ))
        .all()
    )

    result = [
        {
            'latent_point_id': proj.latent_point_id,
            'start_date': proj.point.start_date.isoformat(),
            'x': proj.x,
            'y': proj.y,
            'som_config_id': proj.config_id
        }
        for proj in projections
    ]

    return jsonify(result), 200
