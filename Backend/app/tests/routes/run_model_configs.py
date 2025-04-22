# scripts/run_model_configs.py
"""
Automated CRUD exercise for ModelRunConfig routes using Flask development server.
Usage: python run_model_configs.py
"""
import os
import sys
import time

from Shared.config.dotenv_loader import load_environment
from Backend.app import create_app, db
from Backend.app.models.run_models import ModelRunConfig

# 1. Load environment variables
load_environment()

# 2. Create Flask app in development mode (no TESTING override)
app = create_app()

# 3. Ensure all tables exist
with app.app_context():
    db.create_all()

# 4. Initialize test client
client = app.test_client()
base = '/api/model-configs'

# 5. LIST configs
resp = client.get(base)
print('GET', base, '->', resp.status_code, resp.get_json())

# 6. CREATE a new config
new_cfg = {
    'model_type': 'LSTMVAE',
    'parameters': {'seq_len': 6, 'hidden_size': 64, 'latent_dim': 6},
    'currency_pairs': ['EUR/USD', 'GBP/JPY']
}
resp = client.post(base, json=new_cfg)
print('POST', base, '->', resp.status_code, resp.get_json())
if resp.status_code != 201:
    sys.exit(1)
cfg_id = resp.get_json()['id']

# 7. GET the created config
cfg_url = f"{base}/{cfg_id}"
resp = client.get(cfg_url)
print('GET', cfg_url, '->', resp.status_code, resp.get_json())

# 8. UPDATE the config
update_data = {
    'parameters': {'seq_len': 10, 'hidden_size': 128, 'latent_dim': 8},
    'currency_pairs': ['USD/JPY']
}
resp = client.patch(cfg_url, json=update_data)
print('PATCH', cfg_url, '->', resp.status_code, resp.get_json())

# 9. GET after update
resp = client.get(cfg_url)
print('GET', cfg_url, '->', resp.status_code, resp.get_json())

# 10. TRIGGER training
train_url = f"{cfg_url}/train"
resp = client.post(train_url)
print('POST', train_url, '->', resp.status_code, resp.get_json())

# Allow background training to start
time.sleep(1)

# 11. DELETE the config
resp = client.delete(cfg_url)
print('DELETE', cfg_url, '->', resp.status_code)

# 12. LIST to confirm deletion
resp = client.get(base)
print('GET', base, '->', resp.status_code, resp.get_json())
