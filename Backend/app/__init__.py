from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from Shared.config.dotenv_loader import load_environment
from Shared.config.database import get_database_uri

db = SQLAlchemy()

def create_app():
    load_environment()
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = get_database_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    # register blueprints, etc.
    return app
