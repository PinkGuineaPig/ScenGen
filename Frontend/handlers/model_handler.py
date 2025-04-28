import requests
from dash import callback_context
from Frontend.handlers.base import BaseConfigHandler, API_BASE

class ModelConfigHandler(BaseConfigHandler):
    id_prefix = "model"

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        """
        Handle CRUD actions for model configs.

        Params:
        - trigger: which component fired (e.g. 'model-delete-btn', 'model-confirm-delete', 'model-add-btn')
        - state: dict containing all State values (table_data, selected_rows, model inputs)

        Returns a dict with any of: 'table', 'dialog', 'message', 'clear'.
        """
        table_data    = state.get('table_data', [])
        selected_rows = state.get('selected_rows', [])

        # 1) User clicked "Delete": show confirmation dialog
        if trigger == 'model-delete-btn':
            if not selected_rows:
                return {}
            cfg = table_data[selected_rows[0]]
            return {
                'dialog': True,
                'message': f"Delete model config {cfg['id']}? This can't be undone."
            }

        # 2) User responded to ConfirmDialog
        if trigger == 'model-confirm-delete':
            fired = callback_context.triggered[0]['prop_id']
            # Confirm button
            if fired.endswith('submit_n_clicks') and selected_rows:
                cfg_id = table_data[selected_rows[0]]['id']
                res = requests.delete(f"{API_BASE}/model-configs/{cfg_id}")
                if not res.ok:
                    print(f"🔴 Delete failed for config {cfg_id}: {res.status_code}")
                # update local table
                new_table = [r for r in table_data if r['id'] != cfg_id]
                return {
                    'table': new_table,
                    'dialog': False,
                    'clear': []
                }
            # Cancel button or other → just hide dialog
            return {'dialog': False}

        # 3) Add/Train model
        if trigger == 'model-add-btn':
            # extract inputs
            mtype   = state.get('model_type')
            pairs   = state.get('pairs')
            latent  = state.get('latent_dim')
            base_kl = state.get('base_kl_weight')
            batch   = state.get('batch_size')
            seq_len = state.get('seq_len')
            epochs  = state.get('epochs')
            hidden  = state.get('hidden_size')
            num_layers = state.get('num_layers')
            bidirectional = state.get('bidirectional')
            lr      = state.get('learning_rate')

            # basic validation
            if not all([mtype, pairs, latent, base_kl, batch, seq_len, epochs, hidden, num_layers, lr]):
                return {}

            payload = {
                'model_type':     mtype,
                'currency_pairs': pairs,
                'parameters': {
                    'latent_dim':     latent,
                    'base_kl_weight': base_kl,
                    'batch_size':     batch,
                    'seq_len':        seq_len,
                    'epochs':         epochs,
                    'hidden_size':    hidden,
                    'num_layers':     num_layers,
                    'bidirectional': bool(bidirectional and bidirectional[0]),
                    'learning_rate':  lr
                }
            }
            create = requests.post(f"{API_BASE}/model-configs", json=payload)
            if not create.ok:
                print(f"❌ Create failed: {create.status_code}")
                return {}
            cfg_id = create.json().get('id')

            # trigger training
            train_res = requests.post(f"{API_BASE}/model-configs/{cfg_id}/train")
            if not train_res.ok:
                print(f"❌ Train trigger failed: {train_res.status_code}")

            # refresh table
            try:
                resp = requests.get(f"{API_BASE}/model-configs")
                resp.raise_for_status()
                new_table = resp.json()
            except Exception as e:
                print("❌ Refresh table failed:", e)
                new_table = []

            # auto-select the new config
            new_idx = next((i for i, r in enumerate(new_table) if r.get('id') == cfg_id), None)
            return {
                'table': new_table,
                'clear': [new_idx] if new_idx is not None else []
            }

        # fallback: no changes
        return {}
