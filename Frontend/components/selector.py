# Frontend/components/selector.py

import requests
from dash import html, dcc, callback, Input, Output, State, dash_table

def ConfigTableSection():
    return html.Div([
        # One-time trigger to load data on page load
        dcc.Interval(
            id='load-configs-interval',
            interval=1,
            n_intervals=0,
            max_intervals=1
        ),

        # Table of model configurations
        dash_table.DataTable(
            id='configs-table',
            columns=[
                {'name': col.replace('_', ' ').title(), 'id': col}
                for col in [
                    'id', 'model_type', 'learning_rate', 'latent_dim',
                    'base_kl_weight', 'batch_size', 'seq_len', 'epochs',
                    'hidden_size', 'currency_pairs', 'created_at',
                    'pca_n_components', 'pca_whiten', 'pca_solver',
                    'som_dims', 'som_iterations', 'som_sigma', 'som_learning_rate'
                ]
            ],
            data=[],
            filter_action='native',
            sort_action='native',
            row_selectable='single',
            selected_rows=[],
            page_size=10,
            style_table={'overflowX': 'auto', 'marginTop': '20px'}
        ),

        # Show which row is selected
        html.Div(id='selected-config-id', style={'marginTop': '10px'})
    ], style={'padding': '20px'})


@callback(
    Output('configs-table', 'data'),
    Input('load-configs-interval', 'n_intervals')
)
def load_table_data(n_intervals):
    try:
        resp = requests.get('http://localhost:5000/api/configs/')
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print('Error loading configs:', e)
        return []


@callback(
    Output('selected-config-id', 'children'),
    Input('configs-table', 'selected_rows'),
    State('configs-table', 'data')
)
def display_selected(selected_rows, table_data):
    if selected_rows and table_data:
        row = table_data[selected_rows[0]]
        return f"Selected Config ID: {row['id']}"
    return "No config selected."
