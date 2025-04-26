# Frontend/handlers/plot_handler.py

import requests
from dash import Output, Input, State, callback
import plotly.graph_objects as go
from Frontend.handlers.base import API_BASE

# --- Utility function ---
def empty_figure(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        annotations=[{
            'text': message,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 20}
        }],
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    return fig

# --- Callback to fetch losses and update plots ---
@callback(
    Output('loss-plot', 'figure'),
    Output('kl-plot', 'figure'),
    Input('configs-table', 'selected_rows'),
    State('configs-table', 'data')
)
def update_plots(selected_rows, table_data):
    if not selected_rows or not table_data:
        return empty_figure("No Loss Data Available"), empty_figure("No KL Data Available")

    selected_row = table_data[selected_rows[0]]
    config_id = selected_row.get('id')
    if not config_id:
        return empty_figure("Invalid Config ID"), empty_figure("Invalid Config ID")

    # Fetch latest run_id for this config
    try:
        runs_resp = requests.get(f"{API_BASE}/model-configs/{config_id}/runs")
        runs_resp.raise_for_status()
        runs = runs_resp.json()
        if not runs:
            return empty_figure("No runs available"), empty_figure("No runs available")
        run_id = runs[-1]['id']
    except Exception as e:
        print(f"Failed to fetch runs for config {config_id}: {e}")
        return empty_figure("Error fetching runs"), empty_figure("Error fetching runs")

    # Fetch losses for this run
    try:
        loss_resp = requests.get(f"{API_BASE}/runs/{run_id}/losses")
        loss_resp.raise_for_status()
        losses = loss_resp.json()
    except Exception as e:
        print(f"Failed to fetch losses for run {run_id}: {e}")
        return empty_figure("Error fetching Loss"), empty_figure("Error fetching KL")

    if not losses:
        return empty_figure("No Loss Data Available"), empty_figure("No KL Data Available")

    # Separate train/test losses and KL divergences
    train_loss = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'train']
    test_loss  = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'test']
    train_kl   = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'train_kl']
    test_kl    = [(d['epoch'], d['value']) for d in losses if d['loss_type'] == 'test_kl']

    # Create loss plot
    loss_fig = go.Figure()
    if train_loss:
        epochs, values = zip(*train_loss)
        loss_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Train Loss'))
    if test_loss:
        epochs, values = zip(*test_loss)
        loss_fig.add_trace(go.Scatter(x=epochs, y=values, mode='lines', name='Test Loss'))
    loss_fig.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')

    # Create KL plot
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
    Output('som-heatmap-plot', 'figure'),
    Input('som-projection-store', 'data')
)
def update_som_heatmap(projection_data):

    print('updating heatmap!')

    if not projection_data:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No SOM Data", x=0.5, y=0.5, showarrow=False, font_size=20)],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

    # Find grid size automatically
    xs = [p['x'] for p in projection_data]
    ys = [p['y'] for p in projection_data]
    max_x = max(xs) + 1
    max_y = max(ys) + 1

    # Build count matrix
    grid = [[0 for _ in range(max_x)] for _ in range(max_y)]  # [y][x]

    for p in projection_data:
        grid[p['y']][p['x']] += 1  # careful: y first, then x

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale='Viridis'
        )
    )
    fig.update_layout(
        title="SOM Hit Map",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange='reversed'  # so (0,0) is top-left like usual SOMs
    )
    return fig