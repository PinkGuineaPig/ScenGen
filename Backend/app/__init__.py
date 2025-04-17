# Backend/app/__init__.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # 1) Load env‐file via python‑dotenv here if you use it
    #    (load_dotenv(dotenv_path=".env.development") etc.)

    # 2) Pick the right DB URI
    env = os.getenv("FLASK_ENV", "development")
    if env == "testing":
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SCENGEN_TEST_DB_URL")
    elif env == "production":
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SCENGEN_DB_URL")
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SCENGEN_DEV_DB_URL")

    # 3) Any other config...
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # 4) Initialize extensions
    db.init_app(app)

    # 5) Register blueprints, routes, etc.
    #    from .routes import main_bp
    #    app.register_blueprint(main_bp)

    return app
