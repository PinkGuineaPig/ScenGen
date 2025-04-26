import json
import requests
from Frontend.handlers.base import BaseConfigHandler, fetch_all_configs, API_BASE

class PcaConfigHandler(BaseConfigHandler):
    id_prefix = "pca"

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        table_data     = state['table_data']
        selected_rows  = state['selected_rows']
        n_components   = state.get('pca_n_components')
        whiten_list    = state.get('pca_whiten') or []
        solver         = state.get('pca_solver', 'auto')
        whiten_flag    = bool(whiten_list)

        # 1) Add / Train
        if trigger == 'pca-add-btn':
            if not selected_rows:
                return {}
            # get model_config_id → latest run_id
            cfg_id = table_data[selected_rows[0]]['id']
            runs = requests.get(f"{API_BASE}/model-configs/{cfg_id}/runs").json()
            if not runs:
                return {}
            run_id = runs[-1]['id']

            payload = {
                'n_components':    n_components,
                'additional_params': {
                     'whiten': whiten_flag,
                     'solver': solver
                }
            }
            resp = requests.post(f"{API_BASE}/runs/{run_id}/pca-configs", json=payload)
            if not resp.ok:
                return {}
            pca_id = resp.json()['id']
            # trigger PCA training
            requests.post(f"{API_BASE}/runs/{run_id}/pca-configs/{pca_id}/train")

            # refresh table
            new_table = fetch_all_configs()
            idx = next((i for i,r in enumerate(new_table) if 
                        r.get('pca_n_components')==n_components and
                        r.get('pca_solver')==solver), None)
            return {'table': new_table, 'clear': [idx] if idx is not None else []}

        # 2) Update
        if trigger == 'pca-update-btn' and selected_rows:
            row = table_data[selected_rows[0]]
            cfg_id = row['id']               # this is the model_config_id
            runs = requests.get(f"{API_BASE}/model-configs/{cfg_id}/runs").json()
            run_id = runs[-1]['id']
            # assume you have a pca_config_id column in your row
            pca_cfg_id = row.get('pca_cfg_id')
            payload = {
                'n_components': n_components,
                'additional_params': { 'whiten': whiten_flag, 'solver': solver }
            }
            requests.put(f"{API_BASE}/runs/{run_id}/pca-configs/{pca_cfg_id}", json=payload)
            return {'table': fetch_all_configs()}

        # 3) Delete
        if trigger == 'pca-delete-btn' and selected_rows:
            row = table_data[selected_rows[0]]
            cfg_id = row['id']
            runs = requests.get(f"{API_BASE}/model-configs/{cfg_id}/runs").json()
            run_id = runs[-1]['id']
            pca_cfg_id = row.get('pca_cfg_id')
            requests.delete(f"{API_BASE}/runs/{run_id}/pca-configs/{pca_cfg_id}")
            return {'table': fetch_all_configs(), 'clear': []}

        return {}
