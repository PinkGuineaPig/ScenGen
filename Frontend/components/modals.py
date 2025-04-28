import requests
from dash import html, dcc, callback, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

import json

from Frontend.handlers.base import fetch_all_model_configs
from Frontend.handlers.model_handler import ModelConfigHandler
from Frontend.handlers.som_handler import SomConfigHandler
from Frontend.handlers.pca_handler import PcaConfigHandler

import ast

# List of all config handlers
HANDLERS = [ModelConfigHandler, SomConfigHandler, PcaConfigHandler]

API_BASE = "http://localhost:5000/api"

# ----------------------------------------------------------------------------
# UI: three side-by-side modals for Model, SOM, and PCA configs
# ----------------------------------------------------------------------------
def ConfigModals():
    return html.Div([
        dcc.ConfirmDialog(
            id='model-confirm-delete',
            message=''
        ),
        dcc.Interval(
            id='load-pairs-interval',
            interval=1,
            n_intervals=0,
            max_intervals=1
        ),

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
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': '140px 1fr',
                        'gridRowGap': '10px',
                        'alignItems': 'center'
                    },
                    children=[
                        html.Label('Model Type'),
                        dcc.Input(id='model-type', type='text', placeholder='e.g. LSTMVAE', style={'width': '100%'}),
                        html.Label('Currency Pairs'),
                        dcc.Dropdown(id='model-pairs', multi=True, placeholder='Select currency pairs', style={'width': '100%'}),
                        html.Label('Latent Dimension'),
                        dcc.Input(id='model-latent-dim', type='number', min=1, style={'width': '100%'}),
                        html.Label('Base KL Weight'),
                        dcc.Input(id='model-base-kl-weight', type='number', step=0.0001, min=0, style={'width': '100%'}),
                        html.Label('Batch Size'),
                        dcc.Input(id='model-batch-size', type='number', min=1, style={'width': '100%'}),
                        html.Label('Sequence Length'),
                        dcc.Input(id='model-seq-len', type='number', min=1, style={'width': '100%'}),
                        html.Label('Epochs'),
                        dcc.Input(id='model-epochs', type='number', min=1, style={'width': '100%'}),
                        html.Label('Hidden Size'),
                        dcc.Input(id='model-hidden-size', type='number', min=1, style={'width': '100%'}),
                        html.Label('Num Layers'),
                        dcc.Input(id='model-num-layers', type='number', min=1, style={'width': '100%'}),
                        html.Label('Bidirectional'),
                        dcc.Checklist(
                            options=[{'label': 'Use BiLSTM', 'value': True}],
                            value=[],
                            id='model-bidirectional',
                            inline=True
                        ),
                        html.Label('Learning Rate'),
                        dcc.Input(id='model-learning-rate', type='number', step=0.000001, min=0, style={'width': '100%'}),
                    ]
                ),
                html.Br(),
                html.Div([
                    html.Button('Add/Train', id='model-add-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update', id='model-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete', id='model-delete-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Cancel', id='model-cancel-btn', n_clicks=0)
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
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': '140px 1fr',
                        'gridRowGap': '10px',
                        'alignItems': 'center'
                    },
                    children=[
                        html.Label('X Dimension'),
                        dcc.Input(id='som-x-dim', type='number', min=1, style={'width': '100%'}),
                        html.Label('Y Dimension'),
                        dcc.Input(id='som-y-dim', type='number', min=1, style={'width': '100%'}),
                        html.Label('Iterations'),
                        dcc.Input(id='som-iterations', type='number', min=1, style={'width': '100%'}),
                        html.Label('Additional Params (JSON)'),
                        dcc.Textarea(id='som-additional-params', value='{"sigma": 1.2, "learning_rate": 0.3}', style={'width': '100%', 'height': '80px'})
                    ]
                ),
                html.Br(),
                html.Div([
                    html.Button('Add/Train', id='som-add-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update', id='som-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete', id='som-delete-btn', n_clicks=0)
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
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': '140px 1fr',
                        'gridRowGap': '10px',
                        'alignItems': 'center'
                    },
                    children=[
                        html.Label('N Components'),
                        dcc.Input(id='pca-n-components', type='number', min=1, style={'width': '100%'}),
                        html.Label('Whiten'),
                        dcc.Checklist(
                            options=[{'label': 'Whiten', 'value': True}],
                            value=[],
                            id='pca-whiten',
                            inline=True
                        ),
                        html.Label('Solver'),
                        dcc.Dropdown(
                            id='pca-solver',
                            options=[
                                {'label': 'auto', 'value': 'auto'},
                                {'label': 'full', 'value': 'full'},
                                {'label': 'randomized', 'value': 'randomized'}
                            ],
                            value='auto',
                            style={'width': '100%'}
                        )
                    ]
                ),
                html.Br(),
                html.Div([
                    html.Button('Add/Train', id='pca-add-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Update', id='pca-update-btn', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Delete', id='pca-delete-btn', n_clicks=0)
                ])
            ]
        )
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '20px',
        'backgroundColor': '#f9f9f9'
    })



@callback(
    Output('model-pairs', 'options'),
    Input('load-pairs-interval', 'n_intervals')
)
def load_currency_options(n_intervals):
    try:
        res = requests.get(f"{API_BASE}/currency-pairs")
        res.raise_for_status()
        pairs = res.json()
    except Exception:
        pairs = []
    return [{'label': p, 'value': p} for p in pairs]

def flatten_model_rows(raw_rows):
    """Convert list of model‐dicts into Dash table rows with currency_pairs as a string."""
    out = []
    for m in raw_rows:
        # get the raw currency_pairs value
        cp_raw = m.get('currency_pairs', [])

        # if it's already a list, join it; if it's a str, keep it as-is
        if isinstance(cp_raw, list):
            cp_str = ", ".join(cp_raw)
        elif isinstance(cp_raw, str):
            cp_str = cp_raw
        else:
            cp_str = ""

        p = m.get('parameters', {}) or {}
        out.append({
            'id':             m['id'],
            'model_type':     m['model_type'],
            'created_at':     m['created_at'],
            'currency_pairs': cp_str,
            'learning_rate':  p.get('learning_rate'),
            'latent_dim':     p.get('latent_dim'),
            'base_kl_weight': p.get('base_kl_weight'),
            'batch_size':     p.get('batch_size'),
            'seq_len':        p.get('seq_len'),
            'epochs':         p.get('epochs'),
            'hidden_size':    p.get('hidden_size'),
            'bidirectional':  p.get('bidirectional'),
            'num_layers':     p.get('num_layers'),
        })
    return out

@callback(
    Output('model-configs-table', 'data'),
    Output('model-configs-table', 'selected_rows'),
    Output('model-confirm-delete', 'displayed'),
    Output('model-confirm-delete', 'message'),
    Input('load-model-interval',    'n_intervals'),
    Input('model-add-btn',          'n_clicks'),
    Input('model-delete-btn',       'n_clicks'),
    Input('model-confirm-delete',   'submit_n_clicks'),
    Input('model-confirm-delete',   'cancel_n_clicks'),
    State('model-configs-table',    'data'),
    State('model-configs-table',    'selected_rows'),
    State('model-type',             'value'),
    State('model-pairs',            'value'),
    State('model-latent-dim',       'value'),
    State('model-base-kl-weight',   'value'),
    State('model-batch-size',       'value'),
    State('model-seq-len',          'value'),
    State('model-epochs',           'value'),
    State('model-hidden-size',      'value'),
    State('model-num-layers',       'value'),
    State('model-bidirectional',    'value'),
    State('model-learning-rate',    'value'),
)
def model_dispatch(
    n_load,
    add_click, delete_click, confirm_click, cancel_click,
    table_data, sel_rows,
    mtype, pairs, latent_dim, base_kl_weight, batch_size, seq_len,
    epochs, hidden_size, num_layers, bidirectional, learning_rate
):
    trig = callback_context.triggered[0]['prop_id'].split('.')[0]

    # 1) Initial load: fetch & flatten all configs
    if trig == 'load-model-interval':
        raw = fetch_all_model_configs()
        rows = flatten_model_rows(raw)
        sel = [0] if rows else []
        return rows, sel, False, ""

    # 2) Any modal action: Add / Update / Delete
    if trig.startswith('model-'):
        result = ModelConfigHandler.handle(
            trigger        = trig,
            table_data     = table_data,
            selected_rows  = sel_rows,
            model_type     = mtype,
            pairs          = pairs,
            latent_dim     = latent_dim,
            base_kl_weight = base_kl_weight,
            batch_size     = batch_size,
            seq_len        = seq_len,
            epochs         = epochs,
            hidden_size    = hidden_size,
            num_layers     = num_layers,
            bidirectional  = bidirectional,
            learning_rate  = learning_rate
        ) or {}

        new_table = result.get('table', no_update)

        # Only flatten if it's raw API dicts (they contain 'parameters')
        if isinstance(new_table, list) and new_table and 'parameters' in new_table[0]:
            new_table = flatten_model_rows(new_table)

        new_sel   = result.get('clear', no_update)
        dialog    = result.get('dialog', no_update)
        message   = result.get('message', "")

        return new_table, new_sel, dialog, message

    # 3) Otherwise, do nothing
    raise PreventUpdate



# ----------------------------------------------------------------------------
# SOM CRUD dispatcher + reload on model-select
# ----------------------------------------------------------------------------
@callback(
    Output('som-configs-table',    'data'),
    Output('som-configs-table',    'selected_rows'),
    Input('model-configs-table',   'selected_rows'),
    Input('som-add-btn',           'n_clicks'),
    Input('som-update-btn',        'n_clicks'),
    Input('som-delete-btn',        'n_clicks'),
    State('model-configs-table',   'data'),
    State('som-configs-table',     'data'),
    State('som-configs-table',     'selected_rows'),
    State('som-x-dim',             'value'),
    State('som-y-dim',             'value'),
    State('som-iterations',        'value'),
    State('som-additional-params', 'value'),
)
def som_dispatch(
    model_sel, add_click, upd_click, del_click,
    model_rows, som_rows, som_sel,
    x_dim, y_dim, iterations, raw_params
):
    trig = callback_context.triggered[0]['prop_id'].split('.')[0]

    # — inline “flatten & stringify” helper —
    def flatten(rows):
        out = []
        for c in rows:
            # 1) get the dict (or empty dict)
            params = c.get('additional_params') or {}

            # 2) if somehow it’s still a string, parse it back into a dict
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = ast.literal_eval(params)

            # 3) re‐stringify for display
            params_str = json.dumps(params)

            out.append({
                'id':                c['id'],
                'x_dim':             c['x_dim'],
                'y_dim':             c['y_dim'],
                'iterations':        c['iterations'],
                'additional_params': params_str,   # <— string for table
                'created_at':        c.get('created_at')
            })
        return out

    # nothing to do until you have at least one model
    if not model_rows:
        raise PreventUpdate

    # 1) model selection changed → reload + flatten
    if trig == 'model-configs-table':
        cfg_id = model_rows[model_sel[0]]['id'] if model_sel else None
        if not cfg_id:
            return [], []
        resp = requests.get(f"{API_BASE}/model-configs/{cfg_id}/som-configs")
        resp.raise_for_status()
        return flatten(resp.json()), [0] if resp.json() else []

    # 2) SOM add/update/delete
    cfg_id = model_rows[model_sel[0]]['id']

    # parse the user‐entered JSON from the textarea
    params = raw_params or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = ast.literal_eval(params)

    result = SomConfigHandler.handle(
        trigger               = trig,
        table_data            = som_rows or [],
        selected_rows         = som_sel or [],
        model_cfg_id          = cfg_id,
        som_x_dim             = x_dim,
        som_y_dim             = y_dim,
        som_iterations        = iterations,
        som_additional_params = params
    ) or {}

    # get the new list back, flatten if it’s a fresh list
    new_table = result.get('table', som_rows)
    if isinstance(new_table, list):
        new_table = flatten(new_table)

    new_sel = result.get('clear', som_sel)
    return new_table, new_sel


# ----------------------------------------------------------------------------
# PCA CRUD dispatcher + reload on model-select
# ----------------------------------------------------------------------------
@callback(
    Output('pca-configs-table',    'data'),
    Output('pca-configs-table',    'selected_rows'),
    Input('model-configs-table',   'selected_rows'),
    Input('pca-add-btn',           'n_clicks'),
    Input('pca-update-btn',        'n_clicks'),
    Input('pca-delete-btn',        'n_clicks'),
    State('model-configs-table',   'data'),
    State('pca-configs-table',     'data'),
    State('pca-configs-table',     'selected_rows'),
    State('pca-n-components',      'value'),
    State('pca-whiten',            'value'),
    State('pca-solver',            'value'),
)
def pca_dispatch(
    model_sel, add_click, upd_click, del_click,
    model_rows, pca_rows, pca_sel,
    n_components, whiten_input, solver_input
):
    trig = callback_context.triggered[0]['prop_id'].split('.')[0]

    # ─── inline flatten helper ────────────────────────
    def flatten(rows):
        out = []
        for c in rows:
            params = c.get('additional_params') or {}
            # parse if it sneaks in as a string
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = ast.literal_eval(params)
            out.append({
                'id':           c['id'],
                'n_components': c.get('n_components'),
                'whiten':       params.get('whiten', False),
                'svd_solver':   params.get('svd_solver', params.get('solver', '')),
                'created_at':   c.get('created_at')
            })
        return out

    # nothing to do until a model is loaded
    if not model_rows:
        raise PreventUpdate

    # 1) model selection changed → reload & flatten
    if trig == 'model-configs-table':
        cfg_id = model_rows[model_sel[0]]['id'] if model_sel else None
        if not cfg_id:
            return [], []
        resp = requests.get(f"{API_BASE}/model-configs/{cfg_id}/pca-configs")
        resp.raise_for_status()
        raw_list = resp.json()
        rows = flatten(raw_list)
        return rows, [0] if rows else []

    # 2) PCA add/update/delete → call handler, then flatten its table
    cfg_id = model_rows[model_sel[0]]['id']
    result = PcaConfigHandler.handle(
        trigger        = trig,
        table_data     = pca_rows or [],
        selected_rows  = pca_sel or [],
        model_cfg_id   = cfg_id,
        pca_n_components = n_components,
        pca_whiten       = bool(whiten_input),
        pca_solver       = solver_input
    ) or {}

    new_table = result.get('table', pca_rows)
    if isinstance(new_table, list):
        new_table = flatten(new_table)

    new_sel = result.get('clear', pca_sel)
    return new_table, new_sel



@callback(
    Output('model-type',            'value'),
    Output('model-pairs',           'value'),
    Output('model-latent-dim',      'value'),
    Output('model-base-kl-weight',  'value'),
    Output('model-batch-size',      'value'),
    Output('model-seq-len',         'value'),
    Output('model-epochs',          'value'),
    Output('model-hidden-size',     'value'),
    Output('model-num-layers',      'value'),
    Output('model-bidirectional',   'value'),
    Output('model-learning-rate',   'value'),
    Input('model-configs-table',    'selected_rows'),
    State('model-configs-table',    'data'),
    prevent_initial_call=True
)
def populate_model_modal(selected_rows, table_data):
    if not selected_rows or not table_data:
        raise PreventUpdate

    row = table_data[selected_rows[0]]
    print(row.get('currency_pairs'))
    # pull fields out of the selected row
    model_type    = row.get('model_type')
    pairs_raw = row.get('currency_pairs', '')
    pairs         = pairs = [p.strip() for p in pairs_raw.split(',') if p.strip()]
    latent_dim    = row.get('latent_dim')
    base_kl       = row.get('base_kl_weight')
    batch_size    = row.get('batch_size')
    seq_len       = row.get('seq_len')
    epochs        = row.get('epochs')
    hidden_size   = row.get('hidden_size')
    num_layers    = row.get('num_layers')
    bidirectional = [True] if row.get('bidirectional') else []
    learning_rate = row.get('learning_rate')

    return (
        model_type,
        pairs,
        latent_dim,
        base_kl,
        batch_size,
        seq_len,
        epochs,
        hidden_size,
        num_layers,
        bidirectional,
        learning_rate
    )



# ----------------------------------------------------------------------------
# Populate SOM‐modal inputs when a SOM row is selected
# ----------------------------------------------------------------------------
@callback(
    Output('som-x-dim',             'value'),
    Output('som-y-dim',             'value'),
    Output('som-iterations',        'value'),
    Output('som-additional-params', 'value'),
    Input('som-configs-table',      'selected_rows'),
    State('som-configs-table',      'data'),
    prevent_initial_call=True
)
def populate_som_modal(selected, rows):
    if not selected or not rows:
        raise PreventUpdate

    row = rows[selected[0]]
    # Pull the stored fields
    x = row.get('x_dim')
    y = row.get('y_dim')
    iters = row.get('iterations')
    # We stored a JSON blob in additional_params, pass it back as a string
    params = row.get('additional_params') or {}
    params_str = json.dumps(params, indent=2)

    return x, y, iters, params_str

# ----------------------------------------------------------------------------
# Populate PCA‐modal inputs when a PCA row is selected
# ----------------------------------------------------------------------------
@callback(
    Output('pca-n-components', 'value'),
    Output('pca-whiten',       'value'),
    Output('pca-solver',       'value'),
    Input('pca-configs-table', 'selected_rows'),
    State('pca-configs-table', 'data'),
)
def populate_pca_modal(selected_rows, table_data):
    # If nothing is selected, don’t change the modal
    if not selected_rows:
        raise PreventUpdate

    # Grab the dict for the selected row
    row = table_data[selected_rows[0]]

    # n_components is stored as an integer
    n_comp = row.get('n_components')

    # whiten was flattened into a boolean; Checklist wants a list
    whiten = [True] if row.get('whiten') else []

    # svd_solver is stored as a string
    solver = row.get('svd_solver')

    return n_comp, whiten, solver