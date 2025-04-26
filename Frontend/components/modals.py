import requests
from dash import html, dcc, callback, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

import json

from Frontend.handlers.base import fetch_all_configs
from Frontend.handlers.model_handler import ModelConfigHandler
from Frontend.handlers.som_handler import SomConfigHandler
from Frontend.handlers.pca_handler import PcaConfigHandler

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
                        dcc.Input(id='model-base-kl-weight', type='number', step=0.001, min=0, style={'width': '100%'}),
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
                        dcc.Input(id='model-learning-rate', type='number', step=0.0001, min=0, style={'width': '100%'}),
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

@callback(
    Output('configs-table', 'data'),
    Output('model-confirm-delete', 'displayed'),
    Output('model-confirm-delete', 'message'),
    Output('configs-table', 'selected_rows'),
    Input('load-configs-interval', 'n_intervals'),
    Input('model-delete-btn', 'n_clicks'),
    Input('model-confirm-delete', 'submit_n_clicks'),
    Input('model-confirm-delete', 'cancel_n_clicks'),
    Input('model-add-btn', 'n_clicks'),
    Input('som-add-btn', 'n_clicks'),
    Input('som-update-btn', 'n_clicks'),
    Input('som-delete-btn', 'n_clicks'),
    Input('pca-add-btn', 'n_clicks'),
    Input('pca-update-btn', 'n_clicks'),
    Input('pca-delete-btn', 'n_clicks'),
    State('configs-table', 'data'),
    State('configs-table', 'selected_rows'),
    State('model-type', 'value'),
    State('model-pairs', 'value'),
    State('model-latent-dim', 'value'),
    State('model-base-kl-weight', 'value'),
    State('model-batch-size', 'value'),
    State('model-seq-len', 'value'),
    State('model-epochs', 'value'),
    State('model-hidden-size', 'value'),
    State('model-num-layers', 'value'),
    State('model-bidirectional', 'value'),
    State('model-learning-rate', 'value'),
    State('som-x-dim', 'value'),
    State('som-y-dim', 'value'),
    State('som-iterations', 'value'),
    State('som-additional-params', 'value'),
    State('pca-n-components', 'value'),
    State('pca-whiten', 'value'),
    State('pca-solver', 'value')
)
def dispatch_all(
    n_load,
    m_del, m_ok, m_cancel, m_add,
    s_add, s_upd, s_del,
    p_add, p_upd, p_del,
    table_data, sel_rows,
    mtype, pairs, latent, base_kl, batch, seq_len, epochs, hidden, num_layers, bidirectional, lr,
    som_x_dim, som_y_dim, som_iterations, som_additional_params,
    pca_n_components, pca_whiten, pca_solver
):
    trig = callback_context.triggered[0]['prop_id'].split('.')[0]

    # 1) initial load
    if trig == 'load-configs-interval':
            table = fetch_all_configs()
            return table, False, "", [0] if table else []

    # collect all state into a dict for handlers
    state = {
        'table_data': table_data,
        'selected_rows': sel_rows,
        # model params
        'model_type': mtype,
        'pairs': pairs,
        'latent_dim': latent,
        'base_kl_weight': base_kl,
        'batch_size': batch,
        'seq_len': seq_len,
        'epochs': epochs,
        'hidden_size': hidden,
        'num_layers': num_layers,
        'learning_rate': lr,
        'bidirectional': bidirectional,
        # som params
        'som_x_dim': som_x_dim,
        'som_y_dim': som_y_dim,
        'som_iterations': som_iterations,
        'som_additional_params': som_additional_params,
        # PCA:
        'pca_n_components': pca_n_components,
        'pca_whiten':       pca_whiten,
        'pca_solver':       pca_solver
    }

    # 2) dispatch to handlers — dynamically invoke correct handler
    for Handler in HANDLERS:
        if Handler.handles_trigger(trig):
            result = Handler.handle(trigger=trig, **state) or {}
            return (
                result.get('table', no_update),
                result.get('dialog', False),
                result.get('message', ''),
                result.get('clear', no_update)
            )

    # fallback: no change
    return no_update, False, "", no_update

@callback(
    Output('model-type', 'value'),
    Output('model-pairs', 'value'),
    Output('model-latent-dim', 'value'),
    Output('model-base-kl-weight', 'value'),
    Output('model-batch-size', 'value'),
    Output('model-seq-len', 'value'),
    Output('model-epochs', 'value'),
    Output('model-hidden-size', 'value'),
    Output('model-num-layers', 'value'),
    Output('model-bidirectional', 'value'),
    Output('model-learning-rate', 'value'),
    Output('som-x-dim', 'value'),
    Output('som-y-dim', 'value'),
    Output('som-iterations', 'value'),
    Output('som-additional-params', 'value'),
    Output('pca-n-components', 'value'),
    Output('pca-whiten', 'value'),
    Output('pca-solver', 'value'),
    Input('configs-table', 'selected_rows'),
    State('configs-table', 'data'),
    prevent_initial_call=True
)
def populate_modals(selected_rows, table_data):
    if not selected_rows or not table_data:
        raise PreventUpdate

    row = table_data[selected_rows[0]]
    model_type = row.get('model_type')
    pairs = row.get('currency_pairs', '').split(',') if row.get('currency_pairs') else []
    latent_dim = row.get('latent_dim')
    base_kl = row.get('base_kl_weight')
    batch = row.get('batch_size')
    seq_len = row.get('seq_len')
    epochs = row.get('epochs')
    hidden_size = row.get('hidden_size')
    num_layers = row.get('num_layers')
    bidirectional = [True] if row.get('bidirectional') else []
    learning_rate = row.get('learning_rate')

    som_dims = row.get('som_dims') or ''
    try:
        x_str, y_str = som_dims.split('|')
        x_dim = int(x_str)
        y_dim = int(y_str)
    except Exception:
        x_dim = y_dim = None
    som_iterations = row.get('som_iterations')
    som_params = {}
    if row.get('som_sigma') is not None:
        som_params['sigma'] = row['som_sigma']
    if row.get('som_learning_rate') is not None:
        som_params['learning_rate'] = row['som_learning_rate']
    som_json = json.dumps(som_params) if som_params else ''

    pca_n = row.get('pca_n_components')
    pca_whiten = [True] if row.get('pca_whiten') else []
    pca_solver = row.get('pca_solver') or 'auto'

    return (
        model_type, pairs, latent_dim, base_kl, batch,
        seq_len, epochs, hidden_size, num_layers, bidirectional, learning_rate,
        x_dim, y_dim, som_iterations, som_json,
        pca_n, pca_whiten, pca_solver
    )
