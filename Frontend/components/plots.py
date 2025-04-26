# Frontend/components/plots.py

from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import requests
from Frontend.handlers.base import API_BASE

def ConfigPlots():
    return html.Div(
        [
            dcc.Store(id='som-projection-store'),

            html.Div(
                id='plot-section',
                style={
                    'display': 'flex',          # Horizontal flex layout
                    'gap': '20px',               # Space between left and right
                    'marginTop': '40px',
                    'padding': '20px',
                    'backgroundColor': '#f9f9f9',
                    'border': '1px solid #ccc',
                    'borderRadius': '6px',
                    'boxShadow': '2px 2px 8px rgba(0,0,0,0.05)'
                },
                children=[
                    # Left side: Losses stacked
                    html.Div(
                        id='losses-container',
                        style={
                            'flex': '1',
                            'display': 'flex',
                            'flexDirection': 'column',    # Stack vertically
                            'gap': '20px',
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '6px',
                            'border': '1px solid #ddd',
                            'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                        },
                        children=[
                            html.Div([
                                html.H4("Training Loss", style={'textAlign': 'center'}),
                                dcc.Graph(id='loss-plot')
                            ]),
                            html.Div([
                                html.H4("KL Divergence", style={'textAlign': 'center'}),
                                dcc.Graph(id='kl-plot')
                            ]),
                        ]
                    ),

                    # Right side: SOM plot
                    html.Div(
                        id='som-plot-container',
                        style={
                            'flex': '1',
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '6px',
                            'border': '1px solid #ddd',
                            'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                        },
                        children=[
                            html.H4("SOM Heatmap", style={'textAlign': 'center'}),
                            dcc.Graph(id='som-heatmap-plot')
                        ]
                    ),
                ]
            )
        ]
    )

def placeholder_figure(message):
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': message,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 20}
        }]
    )
    return fig

# ----------------------------------------------------------------------------
# Callback to fetch and update plots
# ----------------------------------------------------------------------------
@callback(
    Output('loss-plot', 'figure'),
    Output('kl-plot', 'figure'),
    Input('configs-table', 'selected_rows'),
    State('configs-table', 'data')
)
def update_plots(selected_rows, table_data):
    if not selected_rows or not table_data:
        return placeholder_figure("No Loss Data Available"), placeholder_figure("No KL Data Available")

    selected = table_data[selected_rows[0]]
    config_id = selected.get('id')

    if not config_id:
        return placeholder_figure("No Loss Data Available"), placeholder_figure("No KL Data Available")

    try:
        res = requests.get(f"{API_BASE}/model-runs/{config_id}/losses")
        res.raise_for_status()
        losses = res.json()
    except Exception:
        return placeholder_figure("Error fetching Loss"), placeholder_figure("Error fetching KL")

    if not losses:
        return placeholder_figure("No Loss Data Available"), placeholder_figure("No KL Data Available")

    # Separate train/test losses and KL divergences
    train_loss = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'train']
    test_loss  = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'test']
    train_kl   = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'train_kl']
    test_kl    = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'test_kl']

    loss_fig = go.Figure()
    if train_loss:
        epochs, values = zip(*train_loss)
        loss_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Train Loss'))
    if test_loss:
        epochs, values = zip(*test_loss)
        loss_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Test Loss'))

    loss_fig.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')

    kl_fig = go.Figure()
    if train_kl:
        epochs, values = zip(*train_kl)
        kl_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Train KL'))
    if test_kl:
        epochs, values = zip(*test_kl)
        kl_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Test KL'))

    kl_fig.update_layout(title='KL Divergence Curve', xaxis_title='Epoch', yaxis_title='KL Divergence')

    return loss_fig, kl_fig





@callback(
    Output('som-projection-store', 'data'),
    Input('configs-table', 'selected_rows'),
    State('configs-table', 'data')
)
def fetch_som_projections(selected_rows, table_data):
    if not selected_rows or not table_data:
        return []
    
    config = table_data[selected_rows[0]]
    config_id = config.get('id')
    if not config_id:
        return []
    
    # First: get run_id
    try:
        run_resp = requests.get(f"{API_BASE}/model-configs/{config_id}/runs")
        run_resp.raise_for_status()
        runs = run_resp.json()
        if not runs:
            return []
        run_id = runs[-1]['id']
    except Exception as e:
        print(f"Failed to fetch run: {e}")
        return []

    # Second: get SOM projections
    try:
        som_resp = requests.get(f"{API_BASE}/runs/{run_id}/som-projections")
        som_resp.raise_for_status()
        projections = som_resp.json()
        return projections
    except Exception as e:
        print(f"Failed to fetch SOM projections: {e}")
        return []
    

@callback(
    Output('som-heatmap-plot', 'figure'),
    Input('som-projection-store', 'data')
)
def update_som_heatmap(store_data):
    if not store_data:
        return placeholder_figure("No SOM Projections Available")

    # Step 1: Find max x and y to define grid size
    max_x = max(p['x'] for p in store_data)
    max_y = max(p['y'] for p in store_data)

    # Step 2: Initialize grid
    grid = [[0 for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Step 3: Count points per cell
    for point in store_data:
        x = point['x']
        y = point['y']
        grid[y][x] += 1

    # Step 4: Fresh heatmap
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=False,
        zmin=0,                         # Force scale always from 0
        zmax=max(max(row) for row in grid)  # Maximum density
    ))

    fig.update_layout(
        title="SOM Heatmap",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='y'
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            autorange='reversed'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig
