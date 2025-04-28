import requests
from Frontend.handlers.base import BaseConfigHandler, API_BASE

class PcaConfigHandler(BaseConfigHandler):
    """
    Handler for PCA Configuration actions: Add/Train, Update, Delete.

    Expects State keys:
      - table_data:     list of current PCA‐row dicts
      - selected_rows:  list of selected row indices
      - model_cfg_id:   int, the selected ModelRunConfig ID
      - pca_n_components: int
      - pca_whiten:     list (empty or [True])
      - pca_solver:     str
    """
    id_prefix = "pca"

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        model_cfg_id   = state.get('model_cfg_id')
        table_data     = state.get('table_data', [])
        selected_rows  = state.get('selected_rows', [])
        n_components   = state.get('pca_n_components')
        whiten_flag    = bool(state.get('pca_whiten') or [])
        solver         = state.get('pca_solver', 'auto')

        base_url = f"{API_BASE}/model-configs/{model_cfg_id}/pca-configs"

        def fetch_pcas():
            resp = requests.get(base_url)
            resp.raise_for_status()
            return resp.json()

        # 1) Add & trigger train
        if trigger == 'pca-add-btn' and model_cfg_id:
            if n_components is None:
                return {}

            payload = {
                'n_components':      n_components,
                'additional_params': {
                    'whiten':    whiten_flag,
                    'svd_solver': solver
                }
            }
            create = requests.post(base_url, json=payload)
            if not create.ok:
                return {}

            pca_cfg_id = create.json().get('id')
            # kick off async training
            requests.post(f"{base_url}/{pca_cfg_id}/train")

            # reload list & auto‐select new row
            new_rows = fetch_pcas()
            idx = next((i for i, r in enumerate(new_rows) if r['id']==pca_cfg_id), None)
            return {
                'table': new_rows,
                'clear': [idx] if idx is not None else []
            }

        # 2) Update existing
        if trigger == 'pca-update-btn' and selected_rows:
            pca_cfg_id = table_data[selected_rows[0]]['id']
            payload = {
                'n_components':      n_components,
                'additional_params': {
                    'whiten':    whiten_flag,
                    'svd_solver': solver
                }
            }
            upd = requests.put(f"{base_url}/{pca_cfg_id}", json=payload)
            if not upd.ok:
                return {}
            return {'table': fetch_pcas()}

        # 3) Delete
        if trigger == 'pca-delete-btn' and selected_rows:
            pca_cfg_id = table_data[selected_rows[0]]['id']
            dl = requests.delete(f"{base_url}/{pca_cfg_id}")
            if not dl.ok:
                return {}
            return {
                'table': fetch_pcas(),
                'clear': []
            }

        # otherwise no action
        return {}
