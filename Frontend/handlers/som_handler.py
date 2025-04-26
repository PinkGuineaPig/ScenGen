import json
import requests
from Frontend.handlers.base import BaseConfigHandler, fetch_all_configs, API_BASE

class SomConfigHandler(BaseConfigHandler):
    """
    Handler for SOM Configuration actions: Add/Train, Update, Delete.

    Expects State keys:
      - table_data: list of row dicts
      - selected_rows: list of selected row indices
      - som_x_dim: int
      - som_y_dim: int
      - som_iterations: int
      - som_additional_params: str (JSON)
    """
    id_prefix = "som"

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        table_data    = state.get('table_data', [])
        selected_rows = state.get('selected_rows', [])

        # parse SOM inputs
        x_dim      = state.get('som_x_dim')
        y_dim      = state.get('som_y_dim')
        iterations = state.get('som_iterations')
        raw_json   = state.get('som_additional_params', '')
        try:
            additional_params = json.loads(raw_json) if raw_json else {}
        except json.JSONDecodeError:
            additional_params = {}

        # 1) Add/Train SOM
        if trigger == 'som-add-btn':
            if not selected_rows:
                return {}
            # Derive model_config_id from selected row
            model_cfg_id = table_data[selected_rows[0]]['id']
            # Fetch runs for this model config
            try:
                runs_resp = requests.get(f"{API_BASE}/model-configs/{model_cfg_id}/runs")
                runs_resp.raise_for_status()
                runs = runs_resp.json()
            except Exception:
                return {}
            if not runs:
                return {}
            # Use the latest run
            run_id = runs[-1]['id']

            payload = {
                'x_dim': x_dim,
                'y_dim': y_dim,
                'iterations': iterations,
                'additional_params': additional_params
            }
            # Create SOM config under the run
            create = requests.post(
                f"{API_BASE}/runs/{run_id}/som-configs", json=payload
            )
            if not create.ok:
                return {}
            som_cfg_id = create.json().get('id')
            # Trigger asynchronous training
            requests.post(
                f"{API_BASE}/runs/{run_id}/som-configs/{som_cfg_id}/train"
            )
            # Refresh table and auto-select new SOM config
            new_table = fetch_all_configs()
            new_idx = next(
                (i for i, r in enumerate(new_table) if r.get('som_cfg_id') == som_cfg_id),
                None
            )
            return {
                'table': new_table,
                'clear': [new_idx] if new_idx is not None else []
            }

        # 2) Update SOM config
        if trigger == 'som-update-btn' and selected_rows:
            model_cfg_id = table_data[selected_rows[0]]['id']
            # Assuming SOM config id is stored under 'som_cfg_id' in table
            som_cfg_id = table_data[selected_rows[0]].get('som_cfg_id')
            payload = {}
            if x_dim is not None:
                payload['x_dim'] = x_dim
            if y_dim is not None:
                payload['y_dim'] = y_dim
            if iterations is not None:
                payload['iterations'] = iterations
            payload['additional_params'] = additional_params

            update = requests.put(
                f"{API_BASE}/runs/{model_cfg_id}/som-configs/{som_cfg_id}",
                json=payload
            )
            if not update.ok:
                return {}
            return {'table': fetch_all_configs()}

        # 3) Delete SOM config
        if trigger == 'som-delete-btn' and selected_rows:
            model_cfg_id = table_data[selected_rows[0]]['id']
            som_cfg_id = table_data[selected_rows[0]].get('som_cfg_id')
            delete = requests.delete(
                f"{API_BASE}/runs/{model_cfg_id}/som-configs/{som_cfg_id}"
            )
            if not delete.ok:
                return {}
            return {'table': fetch_all_configs(), 'clear': []}

        # No SOM action
        return {}
