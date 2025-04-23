# Backend/app/routes/configs.py

from flask import Blueprint, jsonify
from Backend.app import db
from Backend.app.models.run_models import ModelRunConfig

configs_bp = Blueprint('configs', __name__, url_prefix='/configs')

@configs_bp.route('/', methods=['GET'])
def list_model_configs():
    """
    Fetch all ModelRunConfigs with their PCA+SOM sub-configs flattened
    into one row per (config × PCA × SOM) and return as JSON.
    """
    rows = ModelRunConfig.all_flat_dicts(db.session)
    return jsonify(rows), 200