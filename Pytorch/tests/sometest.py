#!/usr/bin/env python
import time
import logging
import os
from flask import json
from Backend.app import create_app, db
from Backend.app.models.run_models import ModelRunConfig, ModelRun
import torch
from sqlalchemy import text

# Configure logging to console and file
LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')

# File handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'train_script.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def check_gpu_status():
    """Log GPU availability and details."""
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    return cuda_available

def check_database_status(session):
    """Check database connectivity and data availability."""
    try:
        # Test connection
        session.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        # Check ExchangeRate table for EUR/AUD data
        count = session.execute(text(
            "SELECT COUNT(*) FROM exchange_rate "
            "WHERE base_currency_id IN (SELECT id FROM currency WHERE code = 'EUR') "
            "AND quote_currency_id IN (SELECT id FROM currency WHERE code = 'AUD')"
        )).scalar()
        logger.info(f"Found {count} ExchangeRate rows for EUR/AUD")
        return count > 0
    except Exception as e:
        logger.exception(f"Database check failed: {str(e)}")
        return False

def log_flask_routes(app):
    """Log all registered Flask routes and blueprints for debugging."""
    logger.info("Registered Flask routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"Endpoint: {rule.endpoint}, URL: {rule}, Methods: {rule.methods}")
    logger.info("Registered blueprints:")
    for bp_name, bp in app.blueprints.items():
        logger.info(f"Blueprint: {bp_name}, URL Prefix: {bp.url_prefix or 'None'}")

def create_new_config(client):
    """Create a new ModelRunConfig via POST /api/model-configs."""
    data = {
        "model_type": "vae",
        "parameters": {
            "seq_len": 100,
            "stride": 1,
            "ffill_limit": 1,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "epochs": 15,
            "base_kl_weight": 0.0001,
            "hidden_size": 256,
            "latent_dim": 32,
            "num_layers": 2,
            "test_every": 1
        },
        "currency_pairs": ["EUR/AUD"]
    }
    try:
        logger.info("Attempting POST /api/model-configs...")
        resp = client.post("/api/model-configs", json=data)
        logger.info(f"Response status: {resp.status_code}, body: {resp.data}")
        if resp.status_code in (200, 201):
            body = resp.get_json()
            cfg_id = body.get("id")
            logger.info(f"Created new config with id {cfg_id}")
            return cfg_id
        else:
            logger.error(f"Failed to create config: {resp.status_code} {resp.data}")
            raise RuntimeError("Could not create ModelRunConfig")
    except Exception as e:
        logger.exception(f"Error creating config: {str(e)}")
        raise RuntimeError("Could not create ModelRunConfig")

def check_logs_for_errors():
    """Check training and dataset logs for errors."""
    log_files = [
        os.path.join(LOG_DIR, 'vae_trainer.log'),
        os.path.join(LOG_DIR, 'dataset.log'),
        os.path.join(LOG_DIR, 'model_configs.log')
    ]
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                last_lines = '\n'.join(content.splitlines()[-10:])  # Last 10 lines
                if 'ERROR' in content or 'Exception' in content:
                    logger.warning(f"Found errors in {log_file}:\n{last_lines}")
                else:
                    logger.info(f"No errors in {log_file}, last lines:\n{last_lines}")
        else:
            logger.warning(f"Log file {log_file} not found")

def main():
    DEV_DB = "postgresql://scengen_dev_user:dev_secret@localhost/scengen_dev"
    app = create_app({"SQLALCHEMY_DATABASE_URI": DEV_DB})

    # Log Flask app and blueprint details
    logger.info("Checking Flask app configuration...")
    log_flask_routes(app)

    # Check GPU status
    logger.info("Checking GPU status...")
    if not check_gpu_status():
        logger.warning("CUDA not available, training will use CPU")

    with app.app_context():
        try:
            db.create_all()
            session = db.session

            # Check database status
            logger.info("Checking database status...")
            if not check_database_status(session):
                raise RuntimeError("Database check failed, no EUR/AUD data found")

            client = app.test_client()

            # 1) Create new config
            logger.info("Creating new ModelRunConfig...")
            cfg_id = create_new_config(client)

            # 2) Trigger training via API
            logger.info(f"Triggering training for config {cfg_id}...")
            start_time = time.time()
            trainer_log = os.path.join(LOG_DIR, 'vae_trainer.log')
            dataset_log = os.path.join(LOG_DIR, 'dataset.log')
            last_trainer_size = 0 if not os.path.exists(trainer_log) else os.path.getsize(trainer_log)
            last_dataset_size = 0 if not os.path.exists(dataset_log) else os.path.getsize(dataset_log)
            resp = client.post(f"/api/model-configs/{cfg_id}/train")
            if resp.status_code != 200:
                logger.error("Failed to start training: %s %s", resp.status_code, resp.data)
                check_logs_for_errors()
                raise RuntimeError(f"Training failed: {resp.data}")
            logger.info(f"Training API call completed in {time.time() - start_time:.2f}s")

            # 3) Poll for completion by querying the DB directly
            logger.info("Polling for ModelRun...")
            run_id = None
            for _ in range(120):
                time.sleep(1)
                run = session.query(ModelRun).filter_by(config_id=cfg_id).order_by(ModelRun.id.desc()).first()
                if run and run.model_blob:
                    run_id = run.id
                    break
                check_logs_for_errors()
            if not run_id:
                logger.error("No run found after training in DB")
                check_logs_for_errors()
                raise RuntimeError("No ModelRun found")
            logger.info(f"Detected run.id = {run_id}")

            # 4) Fetch loss history
            logger.info(f"Fetching loss history for run {run_id}...")
            resp = client.get(f"/api/model-runs/{run_id}/losses")
            if resp.status_code != 200:
                logger.error("Failed to fetch losses: %s %s", resp.status_code, resp.data)
                raise RuntimeError("Could not fetch losses")
            losses = resp.get_json()
            logger.info("Loss history: %s", json.dumps(losses, indent=2))

            logger.info("All routes tested successfully")

        except Exception as e:
            logger.exception("Script failed: %s", str(e))
            raise

if __name__ == "__main__":
    main()