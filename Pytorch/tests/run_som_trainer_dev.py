# scripts/run_som_trainer_dev.py
"""
Run only the SOM trainer on the most recent ModelRun's latent points.
Usage: python run_som_trainer_dev.py
"""
import os

# 1. Load development environment variables
from Shared.config.dotenv_loader import load_environment
load_environment()

# 2. Create Flask app and ensure tables exist
from Backend.app import create_app, db
app = create_app()
with app.app_context():
    db.create_all()

# 3. Import models and SOM trainer
from Backend.app.models.run_models import ModelRun
from Pytorch.trainers.som_trainer import run_som_for_run


def main():
    with app.app_context():
        session = db.session

        # 4. Retrieve the latest ModelRun ID
        run_id = session.query(db.func.max(ModelRun.id)).scalar()
        if run_id is None:
            print("No model runs found in the database.")
            return

        print(f"Selected latest run_id={run_id} for SOM training.")

        # 5. Define SOM hyperparameters (customize as needed)
        som_hyperparams = {
            'som_x_dim': 10,
            'som_y_dim': 10,
            'som_iterations': 500,
            'som_sigma': 1.0,
            'som_learning_rate': 0.5
        }

        # 6. Run SOM training on the selected run_id
        plot_dir = os.path.join('som_plots', f"run_{run_id}")
        cfg_id = run_som_for_run(session, run_id, som_hyperparams, plot_dir=plot_dir)
        print(f"SOM training complete (config_id={cfg_id}). Plots saved to {plot_dir}.")


if __name__ == '__main__':
    main()
