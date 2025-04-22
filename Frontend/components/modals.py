# Frontend/components/modals.py
import json
import requests
from dash import html, dcc, callback, Input, Output, State, callback_context

# Modal components for adding/editing Model, SOM and PCA configurations
def ConfigModals():
    return html.Div([
        # Store available currency pairs fetched from backend
        dcc.Store(id='currency-pairs-store'),

        # Confirm deletion dialog
        dcc.ConfirmDialog(
            id='confirm-delete',
            message='Are you sure you want to delete this configuration?'
        ),

        # Model Config Modal
        html.Div(
            id='model-modal',
            style={'display': 'none', 'position': 'fixed', 'top': '20%', 'left': '30%',
                   'width': '40%', 'padding': '20px', 'backgroundColor': 'white', 'border': '1px solid #ccc',
                   'boxShadow': '2px 2px 10px rgba(0,0,0,0.3)', 'zIndex': 1000},
            children=[
                html.H4(id='model-modal-title'),
                html.Div([
                    html.Label('Model Type'),
                    dcc.Input(id='model-type', type='text'), html.Br(), html.Br(),

                    html.Label('Currency Pairs'),
                    dcc.Dropdown(
                        id='model-pairs',
                        multi=True,
                        placeholder='Select one or more currency pairs'
                    ), html.Br(), html.Br(),

                    html.Label('Parameters (JSON)'),
                    dcc.Textarea(id='model-params', style={'width': '100%', 'height': '100px'})
                ], style={'marginBottom': '10px'}),
                html.Button('Save', id='model-save-btn', n_clicks=0),
                html.Button('Cancel', id='model-cancel-btn', n_clicks=0)
            ]
        ),

        # SOM Modal
        html.Div(
            id='som-modal',
            style={'display': 'none', 'position': 'fixed', 'top': '20%', 'left': '30%',
                   'width': '40%', 'padding': '20px', 'backgroundColor': 'white', 'border': '1px solid #ccc',
                   'boxShadow': '2px 2px 10px rgba(0,0,0,0.3)', 'zIndex': 1000},
            children=[
                html.H4(id='som-modal-title'),
                html.Div([
                    html.Label('x_dim'), dcc.Input(id='som-x-dim', type='number', min=1), html.Br(),
                    html.Label('y_dim'), dcc.Input(id='som-y-dim', type='number', min=1), html.Br(),
                    html.Label('iterations'), dcc.Input(id='som-iterations', type='number', min=1), html.Br(),
                    html.Label('sigma'), dcc.Input(id='som-sigma', type='number', step=0.1, min=0), html.Br(),
                    html.Label('learning_rate'), dcc.Input(id='som-lr', type='number', step=0.01, min=0)
                ], style={'marginBottom': '10px'}),
                html.Button('Save', id='som-save-btn', n_clicks=0),
                html.Button('Cancel', id='som-cancel-btn', n_clicks=0)
            ]
        ),

        # PCA Modal with explicit selectors for parameters
        html.Div(
            id='pca-modal',
            style={'display': 'none', 'position': 'fixed', 'top': '20%', 'left': '30%',
                   'width': '40%', 'padding': '20px', 'backgroundColor': 'white', 'border': '1px solid #ccc',
                   'boxShadow': '2px 2px 10px rgba(0,0,0,0.3)', 'zIndex': 1000},
            children=[
                html.H4(id='pca-modal-title'),
                html.Div([
                    html.Label('Number of Components'),
                    dcc.Input(id='pca-components', type='number', min=1), html.Br(), html.Br(),

                    html.Label('Whiten'),
                    dcc.Checklist(
                        id='pca-whiten',
                        options=[{'label': 'True', 'value': True}],
                        value=[]
                    ), html.Br(), html.Br(),

                    html.Label('SVD Solver'),
                    dcc.Dropdown(
                        id='pca-svd-solver',
                        options=[
                            {'label': 'Auto', 'value': 'auto'},
                            {'label': 'Full', 'value': 'full'},
                            {'label': 'Randomized', 'value': 'randomized'},
                            {'label': 'Arpack', 'value': 'arpack'}
                        ],
                        placeholder='Select solver'
                    )
                ], style={'marginBottom': '10px'}),
                html.Button('Save', id='pca-save-btn', n_clicks=0),
                html.Button('Cancel', id='pca-cancel-btn', n_clicks=0)
            ]
        )
    ])

# Populate currency-pairs store when modal opens
@callback(
    Output('currency-pairs-store', 'data'),
    Input('model-modal', 'style')
)
def load_currency_pairs(_):
    # Fetch available pairs from backend
    try:
        res = requests.get('http://localhost:5000/api/currency-pairs')
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []

# Toggle Model Modal visibility and prefill inputs
@callback(
    Output('model-modal', 'style'),
    Output('model-modal-title', 'children'),
    Output('model-type', 'value'),
    Output('model-pairs', 'options'),
    Output('model-pairs', 'value'),
    Output('model-params', 'value'),
    Input('add-model-btn', 'n_clicks'),
    Input('edit-model-btn', 'n_clicks'),
    Input('model-cancel-btn', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('model-configs-store', 'data'),
    State('currency-pairs-store', 'data')
)
def toggle_model_modal(add_click, edit_click, cancel_click, selected_model, model_configs, currency_pairs):
    ctx = callback_context.triggered_id
    # Build dropdown options
    options = [{'label': p, 'value': p} for p in currency_pairs]
    if ctx == 'add-model-btn':
        return ({'display': 'block'}, 'Add Model Config', '', options, [], '')
    if ctx == 'edit-model-btn' and selected_model:
        cfg = next((m for m in model_configs if m['id'] == selected_model), None)
        if cfg:
            return (
                {'display': 'block'},
                f"Edit Model Config {selected_model}",
                cfg['model_type'],
                options,
                cfg['currency_pairs'],
                json.dumps(cfg['parameters'], indent=2)
            )
    # Hide modal
    return ({'display': 'none'}, '', None, options, [], '')
