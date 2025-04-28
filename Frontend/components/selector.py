from dash import html, dcc, callback, Input, Output, State, dash_table
import json
import requests

API_BASE = "http://localhost:5000/api"

def fetch_all_model_configs():
    resp = requests.get(f"{API_BASE}/model-configs")
    resp.raise_for_status()
    return resp.json()

def fetch_som_configs_for_model(model_cfg_id):
    resp = requests.get(f"{API_BASE}/model-configs/{model_cfg_id}/som-configs")
    resp.raise_for_status()
    return resp.json()

def fetch_pca_configs_for_model(model_cfg_id):
    resp = requests.get(f"{API_BASE}/model-configs/{model_cfg_id}/pca-configs")
    resp.raise_for_status()
    return resp.json()

def ConfigTableSection():
    return html.Div([
        # one‐time load on startup
        dcc.Interval(id='load-model-interval', interval=1, n_intervals=0, max_intervals=1),

        html.Div([
            # ─── Model configs ────────────────────────────────────────
            html.H4('Model Configs'),
            dash_table.DataTable(
                id='model-configs-table',
                columns=[
                    {'name': 'ID',             'id': 'id'},
                    {'name': 'Type',           'id': 'model_type'},
                    {'name': 'Created',        'id': 'created_at'},
                    {'name': 'Currency Pairs', 'id': 'currency_pairs'},
                    {'name': 'Learning Rate',  'id': 'learning_rate'},
                    {'name': 'Latent Dim',     'id': 'latent_dim'},
                    {'name': 'KL Weight',      'id': 'base_kl_weight'},
                    {'name': 'Batch Size',     'id': 'batch_size'},
                    {'name': 'Seq Len',        'id': 'seq_len'},
                    {'name': 'Epochs',         'id': 'epochs'},
                    {'name': 'Hidden Size',    'id': 'hidden_size'},
                    {'name': 'Bidirectional',  'id': 'bidirectional'},
                    {'name': 'Num Layers',     'id': 'num_layers'},
                ],
                data=[],
                row_selectable='single',
                selected_rows=[],
                page_size=5,
                filter_action='native',
                sort_action='native',
                style_table={'overflowX': 'auto'}
            ),
            html.Div(id='selected-model-id', style={'marginTop': '10px'}),

            # ─── PCA & SOM ───────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H4('PCA Configs'),
                    dash_table.DataTable(
                        id='pca-configs-table',
                        columns=[
                            {'name': 'ID', 'id': 'id'},
                            {'name': 'Components', 'id': 'n_components'},
                            {'name': 'Whiten', 'id': 'whiten'},
                            {'name': 'Solver', 'id': 'svd_solver'},
                            {'name': 'Created', 'id': 'created_at'},
                        ],
                        data=[], row_selectable='single', selected_rows=[],
                        page_size=5, filter_action='native', sort_action='native',
                        style_table={'overflowX': 'auto'}
                    ),
                    html.Div(id='selected-pca-id', style={'marginTop': '10px'})
                ], style={'flex': 1, 'padding': '5px'}),

                html.Div([
                    html.H4('SOM Configs'),
                    dash_table.DataTable(
                        id='som-configs-table',
                        columns=[
                            {'name': 'ID', 'id': 'id'},
                            {'name': 'X Dim', 'id': 'x_dim'},
                            {'name': 'Y Dim', 'id': 'y_dim'},
                            {'name': 'Iters', 'id': 'iterations'},
                            {'name': 'Params', 'id': 'additional_params'},
                            {'name': 'Created', 'id': 'created_at'},
                        ],
                        data=[], row_selectable='single', selected_rows=[],
                        page_size=5, filter_action='native', sort_action='native',
                        style_table={'overflowX': 'auto'}
                    ),
                    html.Div(id='selected-som-id', style={'marginTop': '10px'})
                ], style={'flex': 1, 'padding': '5px'}),
            ], style={'display': 'flex', 'marginTop': '10px', 'gap': '10px'}),
        ], style={
            'padding': '20px',
            'border': '1px solid #ccc',
            'borderRadius': '6px',
            'backgroundColor': '#fff',
            'width': '100%',
            'margin': '0 auto'
        })
    ], style={'padding': '20px'})

# ───────────────────────────────────────────────────────────────────────────────
# CALLBACKS


# 3) Show which model is selected
@callback(
    Output('selected-model-id', 'children'),
    Input('model-configs-table', 'selected_rows'),
    State('model-configs-table', 'data'),
)
def show_selected_model(sel, rows):
    if not sel or not rows:
        return ""
    return f"Selected Model ID: {rows[sel[0]]['id']}"


@callback(
    Output('selected-pca-id', 'children'),
    Input('pca-configs-table', 'selected_rows'),
    State('pca-configs-table', 'data')
)
def show_pca(sel, rows):
    return f"PCA ID: {rows[sel[0]]['id']}" if sel else ""

@callback(
    Output('selected-som-id', 'children'),
    Input('som-configs-table', 'selected_rows'),
    State('som-configs-table', 'data')
)
def show_som(sel, rows):
    return f"SOM ID: {rows[sel[0]]['id']}" if sel else ""
