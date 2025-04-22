# Backend/app/__init__.py

import os
from flask import Flask, Blueprint
from flask_sqlalchemy import SQLAlchemy
from Shared.config.dotenv_loader import load_environment
from Shared.config.database import get_database_uri

db = SQLAlchemy()

def create_app(config_overrides=None):
    load_environment()
    app = Flask(__name__, instance_relative_config=False)

    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        SQLALCHEMY_DATABASE_URI=get_database_uri(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    if config_overrides:
        app.config.update(config_overrides)

    db.init_app(app)

    # Create one “/api” blueprint
    api_bp = Blueprint('api', __name__, url_prefix='/api')

    # Import and mount all your resource blueprints under /api
    from Backend.app.routes.model_configs import model_configs_bp
    api_bp.register_blueprint(model_configs_bp)       # prefix comes from that bp

    from Backend.app.routes.som_configs import som_bp
    api_bp.register_blueprint(som_bp)                 # som_bp.url_prefix = '/runs/<…>/som-configs'

    from Backend.app.routes.pca_configs import pca_bp
    api_bp.register_blueprint(pca_bp)

    # (Later) you can add model_runs, feedback, etc.
    # from Backend.app.routes.model_runs import model_runs_bp
    # api_bp.register_blueprint(model_runs_bp)

    # Finally, register the single /api blueprint on the app
    app.register_blueprint(api_bp)

    return app
