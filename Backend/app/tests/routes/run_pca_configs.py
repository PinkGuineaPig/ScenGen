# Backend/app/tests/routes/run_pca_configs.py
"""
Automated exercise of PCA config CRUD routes using Flask test client.
Usage: python run_pca_configs.py
"""
import json
from Shared.config.dotenv_loader import load_environment
from Backend.app import create_app, db
from Backend.app.models.run_models import ModelRun, ModelRunConfig

# 1. Load env and create app
load_environment()
app = create_app()

# 2. Ensure tables exist and find (or create) a run + config
with app.app_context():
    db.create_all()

    # look for the most‐recent run
    latest_run = db.session.query(ModelRun).order_by(ModelRun.id.desc()).first()
    if latest_run is None:
        # no run: seed a dummy config + run so tests can proceed
        cfg = ModelRunConfig(
            model_type     = 'TEST',
            parameters     = {},
            currency_pairs = []
        )
        db.session.add(cfg)
        db.session.flush()  # get cfg.id
        
        run = ModelRun(
            config_id  = cfg.id,
            version    = 1,
            model_blob = b''  # just a placeholder
        )
        db.session.add(run)
        db.session.commit()

        run_id = run.id
        print(f"Seeded dummy config id={cfg.id} and run id={run_id}")
    else:
        run_id = latest_run.id
        print(f"Using existing run_id={run_id}")

# 3. Use test client to exercise PCA CRUD
client = app.test_client()
base   = f"/api/runs/{run_id}/pca-configs"

# 4. LIST existing configs
resp = client.get(base)
print("GET", base, "->", resp.status_code, resp.get_json())

# 5. CREATE a new config
new_cfg = {
    "n_components":     4,
    "additional_params":{"whiten": True, "svd_solver": "randomized"}
}
resp = client.post(base, json=new_cfg)
print("POST", base, "->", resp.status_code, resp.get_json())
cfg_id = resp.get_json().get("id")

# 6. GET the created config
cfg_url = f"{base}/{cfg_id}"
resp    = client.get(cfg_url)
print("GET", cfg_url, "->", resp.status_code, resp.get_json())

# 7. UPDATE the config
update_data = {
    "n_components":     3,
    "additional_params":{"whiten": True, "svd_solver": "randomized"}
}
resp = client.patch(cfg_url, json=update_data)
print("PATCH", cfg_url, "->", resp.status_code, resp.get_json())

# 8. GET after update
resp = client.get(cfg_url)
print("GET", cfg_url, "->", resp.status_code, resp.get_json())

# 9. DELETE the config
#resp = client.delete(cfg_url)
#print("DELETE", cfg_url, "->", resp.status_code)

# 10. LIST to confirm deletion
#resp = client.get(base)
#print("GET", base, "->", resp.status_code, resp.get_json())
