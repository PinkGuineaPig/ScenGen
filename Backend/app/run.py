# Backend/app/run.py
from . import create_app, db
import Backend.app.models.run_models
import Backend.app.models.latent_models

def main():
    app = create_app()
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
