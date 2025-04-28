# Frontend/handlers/plot_handler.py

import requests
from dash import Output, Input, State, callback
import plotly.graph_objects as go
from collections import defaultdict

API_BASE = "http://localhost:5000/api"


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


# --- Loss & KL plots for selected model run ---
@callback(
    Output('loss-plot', 'figure'),
    Output('kl-plot',   'figure'),
    Input('model-configs-table', 'selected_rows'),
    State('model-configs-table', 'data'),
)
def update_plots(selected_rows, model_rows):
    if not selected_rows or not model_rows:
        return (
            empty_figure("No Loss Data Available"),
            empty_figure("No KL Data Available")
        )

    cfg = model_rows[selected_rows[0]]
    cfg_id = cfg.get('id')
    if not cfg_id:
        return (
            empty_figure("Invalid Config ID"),
            empty_figure("Invalid Config ID")
        )

    # 1) get latest run for config
    try:
        runs = requests.get(f"{API_BASE}/model-configs/{cfg_id}/runs").json()
        if not runs:
            return (
                empty_figure("No runs available"),
                empty_figure("No runs available")
            )
        run_id = runs[-1]['id']
    except Exception:
        return (
            empty_figure("Error fetching runs"),
            empty_figure("Error fetching runs")
        )

    # 2) fetch losses
    try:
        losses = requests.get(f"{API_BASE}/runs/{run_id}/losses").json()
    except Exception:
        return (
            empty_figure("Error fetching Loss"),
            empty_figure("Error fetching KL")
        )
    if not losses:
        return (
            empty_figure("No Loss Data Available"),
            empty_figure("No KL Data Available")
        )

    # 3) split out curves
    train_loss = [(d['epoch'], d['value']) for d in losses if d['loss_type']=='train']
    test_loss  = [(d['epoch'], d['value']) for d in losses if d['loss_type']=='test']
    train_kl   = [(d['epoch'], d['value']) for d in losses if d['loss_type']=='train_kl']
    test_kl    = [(d['epoch'], d['value']) for d in losses if d['loss_type']=='test_kl']

    # 4) build figures
    loss_fig = go.Figure()
    if train_loss:
        x,y = zip(*train_loss); loss_fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name='Train Loss'))
    if test_loss:
        x,y = zip(*test_loss);  loss_fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name='Test Loss'))
    loss_fig.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')

    kl_fig = go.Figure()
    if train_kl:
        x,y = zip(*train_kl); kl_fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name='Train KL'))
    if test_kl:
        x,y = zip(*test_kl);  kl_fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name='Test KL'))
    kl_fig.update_layout(title='KL Divergence Curve', xaxis_title='Epoch', yaxis_title='KL Divergence')

    return loss_fig, kl_fig


# --- SOM heatmap for selected SOM config ---
@callback(
    Output('som-heatmap-plot', 'figure'),
    Input('som-configs-table', 'selected_rows'),
    State('som-configs-table', 'data'),
)
def update_som_heatmap(selected_rows, som_rows):
    if not selected_rows or not som_rows:
        return empty_figure("No SOM Data")

    cfg = som_rows[selected_rows[0]]
    som_cfg_id = cfg.get('id')
    if not som_cfg_id:
        return empty_figure("Invalid SOM Config")

    # fetch projections for this SOM config
    try:
        proj = requests.get(f"{API_BASE}/som-projections/{som_cfg_id}").json()
    except Exception:
        return empty_figure("Error fetching SOM data")

    if not proj:
        return empty_figure("No SOM Data")

    # 1) determine grid dims
    xs = [p['x'] for p in proj]
    ys = [p['y'] for p in proj]
    max_x, max_y = max(xs) + 1, max(ys) + 1

    # 2) count hits and collect (id, date) per cell
    grid = [[0] * max_x for _ in range(max_y)]
    cell_info = defaultdict(list)
    for p in proj:
        y, x = p['y'], p['x']
        grid[y][x] += 1
        cell_info[(y, x)].append((p['latent_point_id'], p['start_date']))

    # 3) build hover‐text matrix
    hovertext = []
    for row in range(max_y):
        ht_row = []
        for col in range(max_x):
            entries = cell_info.get((row, col), [])
            if entries:
                # e.g. "Count: 3<br>42 (2025-01-15)<br>63 (2025-02-01)<br>..." 
                lines = [f"{pid} ({dt})" for pid, dt in entries]
                ht = f"Count: {len(entries)}<br>" + "<br>".join(lines)
            else:
                ht = "No points"
            ht_row.append(ht)
        hovertext.append(ht_row)

    # 4) render the heatmap with custom hover text
    fig = go.Figure(go.Heatmap(
        z=grid,
        text=hovertext,
        hoverinfo='text'     # show only our hover‐text
    ))
    fig.update_layout(
        title="SOM Hit Map",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_autorange='reversed'
    )
    return fig
