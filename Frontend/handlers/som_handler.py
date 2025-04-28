import json
import requests
from Frontend.handlers.base import BaseConfigHandler, API_BASE

class SomConfigHandler(BaseConfigHandler):
    id_prefix = "som"

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        table_data    = state.get('table_data', [])
        selected_rows = state.get('selected_rows', [])
        model_cfg_id  = state.get('model_cfg_id')  # however you’re passing it in

        # parse additional_params: accept both dicts and JSON strings
        raw = state.get('som_additional_params', {})
        if isinstance(raw, dict):
            additional = raw
        else:
            try:
                additional = json.loads(raw)
            except Exception:
                additional = {}

        x_dim      = state.get('som_x_dim')
        y_dim      = state.get('som_y_dim')
        iterations = state.get('som_iterations')

        # 1) Create & train
        if trigger == 'som-add-btn':
            payload = {
                'x_dim':      x_dim,
                'y_dim':      y_dim,
                'iterations': iterations,
                'additional_params': additional
            }
            resp = requests.post(
                f"{API_BASE}/model-configs/{model_cfg_id}/som-configs",
                json=payload
            )
            if not resp.ok:
                return {}

            new_id = resp.json().get('id')
            # fire off training
            requests.post(
                f"{API_BASE}/model-configs/{model_cfg_id}/som-configs/{new_id}/train"
            )

            # re-fetch table & select new row
            from Frontend.handlers.base import fetch_som_configs
            new_table = fetch_som_configs(model_cfg_id)
            idx = next((i for i,r in enumerate(new_table) if r['id']==new_id), None)
            return {'table': new_table, 'clear': [idx] if idx is not None else []}

        # 2) Update
        if trigger == 'som-update-btn' and selected_rows:
            cfg_id = table_data[selected_rows[0]]['id']
            payload = {'additional_params': additional}
            if x_dim      is not None: payload['x_dim']      = x_dim
            if y_dim      is not None: payload['y_dim']      = y_dim
            if iterations is not None: payload['iterations'] = iterations

            resp = requests.put(
                f"{API_BASE}/model-configs/{model_cfg_id}/som-configs/{cfg_id}",
                json=payload
            )
            if not resp.ok:
                return {}
            from Frontend.handlers.base import fetch_som_configs
            return {'table': fetch_som_configs(model_cfg_id)}

        # 3) Delete
        if trigger == 'som-delete-btn' and selected_rows:
            idx    = selected_rows[0]
            cfg_id = table_data[idx]['id']
            new_table = [r for i,r in enumerate(table_data) if i!=idx]
            requests.delete(
                f"{API_BASE}/model-configs/{model_cfg_id}/som-configs/{cfg_id}"
            )
            return {'table': new_table, 'clear': []}

        return {}
