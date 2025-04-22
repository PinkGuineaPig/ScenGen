# Frontend/components/selector.py
import requests
from dash import html, dcc, callback, Input, Output, State, dash_table

# Section with model selector and CRUD buttons
def SelectorSection():
    return html.Div([
        # Store for model configurations
        dcc.Store(id='model-configs-store'),

        # CRUD controls for ModelRunConfig
        html.Div([
            html.Button('Add Model', id='add-model-btn', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Edit Model', id='edit-model-btn', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Delete Model', id='delete-model-btn', n_clicks=0)
        ], style={'marginBottom': '10px'}),

        # Model selection dropdown
        dcc.Dropdown(
            id='model-dropdown',
            placeholder='Select a model...',
            style={'width': '50%', 'marginBottom': '20px'}
        ),

        # Placeholder for config tables to be rendered
        html.Div(id='config-tables')
    ])

# Callback to load models into dropdown and store
@callback(
    Output('model-dropdown', 'options'),
    Output('model-dropdown', 'value'),
    Output('model-configs-store', 'data'),
    Input('add-model-btn', 'n_clicks'),
    Input('edit-model-btn', 'n_clicks'),
    Input('delete-model-btn', 'n_clicks')
)
def load_models(add_click, edit_click, delete_click):
    """
    Fetch all model configurations from the backend and update dropdown + store.
    """
    try:
        res = requests.get('http://localhost:5000/api/model-configs')
        data = res.json() if res.status_code == 200 else []
    except Exception:
        data = []

    options = [{'label': f"{m['model_type']} (ID {m['id']})", 'value': m['id']} for m in data]
    value = options[0]['value'] if options else None
    return options, value, data

# Callback to render SOM/PCA tables when model selection changes
@callback(
    Output('config-tables', 'children'),
    Input('model-dropdown', 'value'),
    State('model-configs-store', 'data')
)
def render_config_tables(selected_model, model_configs):
    """
    Render placeholder or the actual SOM/PCA tables for the selected model.
    """
    if selected_model is None:
        return html.P('Select a model to view and manage configurations.')

    # Stub for actual tables; replace with real components
    return html.Div([
        html.P(f'Model ID {selected_model} selected. Tables go here.'),
    ])