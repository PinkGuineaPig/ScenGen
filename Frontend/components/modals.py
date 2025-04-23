# Frontend/components/modals.py

import requests
from dash import html, dcc, callback, Input, Output, State, callback_context, no_update

# ----------------------------------------------------------------------------
# ConfigModals: three side-by-side modals for Model, SOM, and PCA configs
# ----------------------------------------------------------------------------
def ConfigModals():
    return html.Div([
        # Confirm dialog for deleting a model
        dcc.ConfirmDialog(
            id='model-confirm-delete',
            message=''  # will be set by callback
        ),

        dcc.Store(id='configs-store'),
        # Store for currency pairs
        dcc.Store(id='currency-pairs-store', storage_type='memory'),
        # Interval to fetch pairs once at startup
        dcc.Interval(id='load-pairs-interval', interval=1, n_intervals=0, max_intervals=1),

        # -------------------------
        # Model Configuration Modal
        html.Div(
            id='model-config-modal',
            style={
                'border': '1px solid #ccc',
                'padding': '20px',
                'width': '30%',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'borderRadius': '4px',
                'backgroundColor': 'white'
            },
            children=[
                html.H4('Model Configuration'),
                html.Label('Model Type'),
                dcc.Input(id='model-type', type='text', placeholder='e.g. LSTMVAE', style={'width': '100%'}),
                html.Br(), html.Br(),
                html.Label('Currency Pairs'),
                dcc.Dropdown(id='model-pairs', multi=True, placeholder='Select currency pairs', style={'width': '100%'}),
                html.Br(), html.Br(),
                html.Label('Latent Dimension'), dcc.Input(id='model-latent-dim', type='number', min=1), html.Br(), html.Br(),
                html.Label('Base KL Weight'), dcc.Input(id='model-base-kl-weight', type='number', step=0.001, min=0), html.Br(), html.Br(),
                html.Label('Batch Size'), dcc.Input(id='model-batch-size', type='number', min=1), html.Br(), html.Br(),
                html.Label('Sequence Length'), dcc.Input(id='model-seq-len', type='number', min=1), html.Br(), html.Br(),
                html.Label('Epochs'), dcc.Input(id='model-epochs', type='number', min=1), html.Br(), html.Br(),
                html.Label('Hidden Size'), dcc.Input(id='model-hidden-size', type='number', min=1), html.Br(), html.Br(),
                html.Label('Learning Rate'), dcc.Input(id='model-learning-rate', type='number', step=0.0001, min=0), html.Br(), html.Br(),
                html.Div([
                    html.Button('Add/Train', id='model-add-btn',    n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update',    id='model-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete',    id='model-delete-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Cancel',    id='model-cancel-btn', n_clicks=0)
                ])
            ]
        ),

        # -------------------------
        # SOM Configuration Modal
        html.Div(
            id='som-config-modal',
            style={
                'border': '1px solid #ccc',
                'padding': '20px',
                'width': '30%',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'borderRadius': '4px',
                'backgroundColor': 'white'
            },
            children=[
                html.H4('SOM Configuration'),
                html.Div([
                    html.Button('Add/Train', id='som-add-btn',    n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update',    id='som-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete',    id='som-delete-btn', n_clicks=0)
                ])
            ]
        ),

        # -------------------------
        # PCA Configuration Modal
        html.Div(
            id='pca-config-modal',
            style={
                'border': '1px solid #ccc',
                'padding': '20px',
                'width': '30%',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'borderRadius': '4px',
                'backgroundColor': 'white'
            },
            children=[
                html.H4('PCA Configuration'),
                html.Div([
                    html.Button('Add/Train', id='pca-add-btn',    n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update',    id='pca-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete',    id='pca-delete-btn', n_clicks=0)
                ])
            ]
        )
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '20px',
        'backgroundColor': '#f9f9f9'
    })


# ----------------------------------------------------------------------------
# Load currency pairs into Store on page load
# ----------------------------------------------------------------------------
@callback(
    Output('currency-pairs-store', 'data'),
    Input('load-pairs-interval', 'n_intervals'),
)
def fetch_currency_pairs(n_intervals):
    """
    Fetch unique currency pairs from backend once when the interval fires.
    """
    try:
        res = requests.get('http://localhost:5000/api/currency-pairs')
        res.raise_for_status()
        return res.json()
    except Exception:
        return []

@callback(
    Output('model-pairs', 'options'),
    Input('currency-pairs-store', 'data')
)
def set_currency_options(pairs):
    print("▶ set_currency_options got:", pairs)
    return [{'label': p, 'value': p} for p in (pairs or [])]


@callback(
    Output('configs-store',        'data'),            # the only writer to configs-store
    Output('model-confirm-delete', 'displayed'),       # show/hide delete dialog
    Output('model-confirm-delete', 'message'),         # delete dialog text
    Output('configs-table',        'selected_rows'),   # clear selection after delete
    Input('load-configs-interval', 'n_intervals'),     # 1) initial load
    Input('model-delete-btn',      'n_clicks'),        # 2) user clicks “Delete”
    Input('model-confirm-delete',  'submit_n_clicks'), # 3a) user confirms delete
    Input('model-confirm-delete',  'cancel_n_clicks'), # 3b) user cancels delete
    Input('model-add-btn',         'n_clicks'),        # 4) user clicks “Add/Train”
    State('configs-table',         'selected_rows'),
    State('configs-table',         'data'),
    State('model-type',            'value'),
    State('model-pairs',           'value'),
    State('model-latent-dim',      'value'),
    State('model-base-kl-weight',  'value'),
    State('model-batch-size',      'value'),
    State('model-seq-len',         'value'),
    State('model-epochs',          'value'),
    State('model-hidden-size',     'value'),
    State('model-learning-rate',   'value'),
    prevent_initial_call=True
)
def manage_configs(
    n_load,
    del_clicks, ok_delete, cancel_delete,
    add_clicks,
    sel_rows, table_data,
    mtype, pairs, latent_dim,
    base_kl, batch_size, seq_len, epochs,
    hidden_size, lr
):
    trg = callback_context.triggered[0]['prop_id'].split('.')[0]

    # 1) Initial page load
    if trg == 'load-configs-interval':
        try:
            return requests.get('http://localhost:5000/api/api/configs').json(), False, '', []
        except:
            return [], False, '', []

    # 2) User clicked “Delete” → show ConfirmDialog
    if trg == 'model-delete-btn':
        if not sel_rows:
            return no_update, False, '', no_update
        cfg_id = table_data[sel_rows[0]]['id']
        msg = (
            f"Are you sure you want to delete model config {cfg_id} "
            "and all related data (loss, PCA, SOM)?"
        )
        return no_update, True, msg, no_update

    # 3) User responded to ConfirmDialog
    if trg == 'model-confirm-delete':
        # 3a) confirmed
        if ok_delete and sel_rows:
            cfg_id = table_data[sel_rows[0]]['id']
            # call your DELETE route
            r = requests.delete(f'http://localhost:5000/api/model-configs/{cfg_id}')
            if not r.ok:
                print(f"Delete failed {cfg_id}: {r.status_code}")
            # remove from local table_data
            new_data = [r for r in table_data if r['id'] != cfg_id]
            return new_data, False, '', []      # hide dialog, clear selection

        # 3b) cancelled
        if cancel_delete:
            return no_update, False, '', no_update

    # 4) User clicked “Add/Train”
    if trg == 'model-add-btn':
        # basic validation
        if not all([mtype, pairs, latent_dim, base_kl, batch_size,
                    seq_len, epochs, hidden_size, lr]):
            return no_update, False, '', no_update

        # 4.1) create config
        payload = {
            "model_type":     mtype,
            "currency_pairs": pairs,
            "parameters": {
                "latent_dim":     latent_dim,
                "base_kl_weight": base_kl,
                "batch_size":     batch_size,
                "seq_len":        seq_len,
                "epochs":         epochs,
                "hidden_size":    hidden_size,
                "learning_rate":  lr
            }
        }

        create = requests.post('http://localhost:5000/api/model-configs', json=payload)
        if not create.ok:
            print("❌ create failed", create.status_code)
            return no_update, False, '', no_update

        cfg_id = create.json().get('id')
        # 4.2) trigger training
        train = requests.post(f'http://localhost:5000/api/model-configs/{cfg_id}/train')
        if not train.ok:
            print("❌ train trigger failed", train.status_code)

        # 4.3) re-fetch full list
        try:
            fresh = requests.get('http://localhost:5000/api/configs').json()
        except:
            fresh = []
        return fresh, False, '', []

    # fallback: do nothing
    return no_update, False, '', no_update
