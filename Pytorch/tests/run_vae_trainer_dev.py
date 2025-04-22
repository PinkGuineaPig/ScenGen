# scripts/run_group_trainer_dev.py
"""
Run `train_model_from_groups` on your development database without modifying or deleting any existing tables or data.
"""
import os

# 1. Load development environment variables
# `.env.development` will be loaded by default (FLASK_ENV defaults to 'development')
from Shared.config.dotenv_loader import load_environment
from Backend.app import create_app, db
load_environment()

from Backend.app import create_app
from Pytorch.trainers.vae_trainer import train_vae_for_config




def main():
    # 2. Create Flask app in development mode
    app = create_app()

    # 3. Ensure all tables exist (recreate missing tables)
    with app.app_context():
        db.create_all()

    # 4. Define currency pairs and hyperparameters
    currency_pairs = ['EUR/USD']  # add more pairs as needed
    hyperparams = {
        'seq_len': 6,
        'hidden_size': 64,
        'latent_dim': 6,
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 1e-3,
        'base_kl_weight': 0.1,
        'test_every': 1
    }

    # 5. Run training on dev database
    print(f"Running training for pairs: {currency_pairs} on development database.")
    train_vae_for_config(app, currency_pairs, hyperparams)
    print("Training run complete.")


if __name__ == '__main__':
    main()

