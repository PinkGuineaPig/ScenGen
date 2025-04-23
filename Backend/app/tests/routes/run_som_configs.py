# Backend/app/tests/routes/run_som_configs.py
"""
Automated exercise of SOM config CRUD routes using Flask test client.
Usage: python run_som_configs.py
"""
import json
from Shared.config.dotenv_loader import load_environment
from Backend.app import create_app, db
from Backend.app.models.run_models import ModelRun

# 1. Load env and create app
load_environment()
app = create_app()

# 2. Ensure tables exist and find latest run_id
with app.app_context():
    db.create_all()
    run_id = db.session.query(db.func.max(ModelRun.id)).scalar()
    if run_id is None:
        print("No ModelRun found. Run the VAE trainer first.")
        exit(1)
    print(f"Using latest run_id={run_id}")

# 3. Use test client to exercise SOM CRUD
client = app.test_client()
base = f"/api/runs/{run_id}/som-configs"

# 4. LIST existing configs
resp = client.get(base)
print("GET", base, "->", resp.status_code, resp.get_json())

# 5. CREATE a new config
new_cfg = {"x_dim": 8, "y_dim": 6, "iterations": 200, "additional_params": {"sigma": 1.2, "learning_rate": 0.3}}
resp = client.post(base, json=new_cfg)
print("POST", base, "->", resp.status_code, resp.get_json())
cfg_id = resp.get_json().get("id")

# 6. GET the created config
cfg_url = f"{base}/{cfg_id}"
resp = client.get(cfg_url)
print("GET", cfg_url, "->", resp.status_code, resp.get_json())

# 7. UPDATE the config
update_data = {"iterations": 300, "additional_params": {"sigma": 0.8}}
resp = client.patch(cfg_url, json=update_data)
print("PATCH", cfg_url, "->", resp.status_code, resp.get_json())

# 8. GET after update
resp = client.get(cfg_url)
print("GET", cfg_url, "->", resp.status_code, resp.get_json())

# 9. DELETE the config
resp = client.delete(cfg_url)
print("DELETE", cfg_url, "->", resp.status_code)

# 10. LIST to confirm deletion
resp = client.get(base)
print("GET", base, "->", resp.status_code, resp.get_json())




base = f"/api/configs/"
resp = client.get(base)
print("GET", base, "->", resp.status_code, resp.get_json())