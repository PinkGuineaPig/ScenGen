# Frontend/components/plots.py

from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import requests
from Frontend.handlers.base import API_BASE
import numpy as np

def ConfigPlots():
    return html.Div(
        [
            dcc.Store(id='som-projection-store'),
            dcc.Store(id='pca-projection-store'),
            dcc.Store(id='pca-explained-store'),

            dcc.Store(id='selected-latent-store'),
            html.Div(id='latent-store-logger', style={'display': 'none'}),  # ← dummy

            html.Div(
                id='plot-section',
                style={
                    'display': 'flex', 'gap': '20px', 'marginTop': '40px',
                    'padding': '20px', 'backgroundColor': '#f9f9f9',
                    'border': '1px solid #ccc', 'borderRadius': '6px',
                    'boxShadow': '2px 2px 8px rgba(0,0,0,0.05)'
                },
                children=[

                    # ─── Left: Loss plots ───────────────────────
                    html.Div(
                        id='losses-container',
                        style={
                            'flex': 1, 'display': 'flex', 'flexDirection': 'column',
                            'gap': '20px', 'backgroundColor': 'white',
                            'padding': '20px', 'border': '1px solid #ddd',
                            'borderRadius': '6px', 'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                        },
                        children=[
                            html.H4("Training & Test Loss", style={'textAlign': 'center'}),
                            dcc.Graph(id='loss-plot'),
                            html.H4("KL Divergence", style={'textAlign': 'center'}),
                            dcc.Graph(id='kl-plot'),
                        ]
                    ),

                    # ─── Right: SOM + PCA ───────────────────────
                    html.Div(
                        style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'},
                        children=[

                            # SOM
                            html.Div(
                                style={
                                    'flex': 1, 'backgroundColor': 'white',
                                    'padding': '20px', 'borderRadius': '6px',
                                    'border': '1px solid #ddd', 'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                                },
                                children=[
                                    html.H4("SOM Heatmap", style={'textAlign': 'center'}),
                                    dcc.Graph(id='som-heatmap-plot',
                                                                  config={
                                                                            'displaylogo': False,      # removes the Plotly logo
                                                                            'displayModeBar': False    # hides the entire mode bar
                                                                            # or use 'modeBarButtonsToRemove' list if you want to keep some buttons
                                                                        },
                                              )
                                ]
                            ),

                            # PCA split
                            html.Div(
                                style={'flex': 1, 'display': 'flex', 'gap': '20px'},
                                children=[

                                    # PCA scatter
                                    html.Div(
                                        style={
                                            'flex': 1, 'backgroundColor': 'white',
                                            'padding': '20px', 'borderRadius': '6px',
                                            'border': '1px solid #ddd', 'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                                        },
                                        children=[
                                            html.H4("PCA: PC1 vs PC2", style={'textAlign': 'center'}),
                                            dcc.Graph(id='pca-scatter-plot')
                                        ]
                                    ),

                                    # PCA elbow
                                    html.Div(
                                        style={
                                            'flex': 1, 'backgroundColor': 'white',
                                            'padding': '20px', 'borderRadius': '6px',
                                            'border': '1px solid #ddd', 'boxShadow': '2px 2px 6px rgba(0,0,0,0.05)'
                                        },
                                        children=[
                                            html.H4("PCA Explained Variance", style={'textAlign': 'center'}),
                                            dcc.Graph(id='pca-elbow-plot')
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
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
    Input('model-configs-table', 'selected_rows'),
    State('model-configs-table', 'data')
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
    Input('som-configs-table', 'selected_rows'),
    State('som-configs-table', 'data')
)
def fetch_som_projections(som_selected_rows, som_table_data):
    # nothing selected or no data
    if not som_selected_rows or not som_table_data:
        return []

    # grab the selected som-config id
    som_row = som_table_data[som_selected_rows[0]]
    som_cfg_id = som_row.get('id')
    if som_cfg_id is None:
        return []

    # fetch the projections list
    url = f"{API_BASE}/som-projections/{som_cfg_id}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        projections = resp.json()
        return projections if isinstance(projections, list) else []
    except Exception as e:
        print(f"Error fetching SOM projections: {e}")
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
        x, y = point['x'], point['y']
        grid[y][x] += 1

    # Step 4: Render heatmap—with hover turned off
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=False,
        zmin=0,
        zmax=max(max(row) for row in grid),
        hoverinfo='none',      # disable hover entirely
        hovertemplate=None     # ensure no hovertemplate is shown
    ))

    fig.update_layout(
        margin=dict(l=1, r=1, t=3, b=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig



# ──────────────────────────────────────────────────────────
# PCA callbacks (new)
# ──────────────────────────────────────────────────────────
@callback(
    Output('pca-projection-store', 'data'),
    Input('pca-configs-table', 'selected_rows'),
    State('pca-configs-table', 'data')
)
def fetch_pca_projections(pca_selected_rows, pca_table_data):
    print('fetch_pca_projections')
    if not pca_selected_rows or not pca_table_data:
        return []
    cfg = pca_table_data[pca_selected_rows[0]]
    cfg_id = cfg.get('id')
    if not cfg_id:
        return []
    try:
        resp = requests.get(f"{API_BASE}/pca-projections/{cfg_id}")
        resp.raise_for_status()
        return resp.json()
    except:
        return []

@callback(
    Output('pca-scatter-plot', 'figure'),
    Input('pca-projection-store', 'data')
)
def update_pca_scatter(proj_data):
    if not proj_data:
        return placeholder_figure("No PCA Data")
    # pivot to PC1 & PC2
    pts = {}
    for p in proj_data:
        pid, dim, val = p['latent_point_id'], p['dim'], p['value']
        pts.setdefault(pid, {})[f"PC{dim+1}"] = val
    rows = [
        {'point': pid, 'PC1': v['PC1'], 'PC2': v['PC2']}
        for pid, v in pts.items() if 'PC1' in v and 'PC2' in v
    ]
    if not rows:
        return placeholder_figure("No PC1/PC2")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r['PC1'] for r in rows],
        y=[r['PC2'] for r in rows],
        mode='markers',
        marker={'size': 8},
        text=[r['point'] for r in rows],
        hovertemplate="Point %{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title='PC1', yaxis_title='PC2',
        margin={'l':10,'r':10,'t':30,'b':10}
    )
    return fig



@callback(
    Output('pca-explained-store', 'data'),
    Input('pca-configs-table', 'selected_rows'),
    State('pca-configs-table', 'data'),
)
def fetch_pca_explained(sel, data):
    print('fetch_pca_explained')
    if not sel or not data:
        return []
    cfg = data[sel[0]]
    cfg_id = cfg.get('id')
    if not cfg_id:
        return []
    try:
        resp = requests.get(f"{API_BASE}/pca-projections/{cfg_id}/explained_variance")
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []
    

@callback(
    Output('pca-elbow-plot', 'figure'),
    Input('pca-explained-store', 'data')
)
def update_pca_elbow(explained):
    if not explained:
        return placeholder_figure("No Explained Variance")
    comps = list(range(1, len(explained)+1))
    cum = np.cumsum(explained)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comps, y=explained, name='Individual'))
    fig.add_trace(go.Scatter(x=comps, y=cum, name='Cumulative', mode='lines+markers', yaxis='y2'))
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Component',
        yaxis=dict(title='Var Ratio'),
        yaxis2=dict(title='Cumulative', overlaying='y', side='right'),
        margin={'l':40,'r':40,'t':40,'b':40}
    )
    return fig






@callback(
    Output('selected-latent-store', 'data'),
    Input('som-heatmap-plot', 'clickData'),
    State('som-projection-store', 'data'),
    prevent_initial_call=True
)
def store_cell_selection(clickData, projections):
    """
    Whenever you click a SOM cell, capture all latent points
    (id + start_date) that fell into that cell, and store them.
    """
    if not clickData or not projections:
        return []

    pt = clickData['points'][0]
    # cast to int so 2.0 → 2
    x_clicked = int(round(pt['x']))
    y_clicked = int(round(pt['y']))

    selected = [
        {
            'latent_point_id': p['latent_point_id'],
            'start_date':      p['start_date']
        }
        for p in projections
        if p['x'] == x_clicked and p['y'] == y_clicked
    ]

    # sanity‐check log
    print(f"Clicked cell ({x_clicked},{y_clicked}) → selected {len(selected)} latent points")

    return selected