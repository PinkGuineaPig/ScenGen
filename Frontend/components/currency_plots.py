# Frontend/components/currency_plots.py
import requests
from dash import html, dcc, callback, Output, Input, State, ALL, MATCH
import plotly.graph_objs as go
from Frontend.handlers.base import API_BASE
import pandas as pd
import dash

def CurrencyPlots():
    return html.Div(
        style={'marginTop': '40px', 'padding': '20px'},
        children=[
            # 1) Store the selected pairs + seq_len
            dcc.Store(id='currency-pair-store'),
            # 2) Store all fetched histories keyed by pair
            dcc.Store(id='currency-history-store'),

            # This container will get one Graph per pair
            html.Div(
                id='currency-plots-container',
                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}
            )
        ]
    )


# Step A: when the model changes, stash pairs+seq_len
@callback(
    Output('currency-pair-store', 'data'),
    Input('model-configs-table', 'selected_rows'),
    State('model-configs-table', 'data'),
    prevent_initial_call=True
)
def _store_pairs_and_seq_len(selected_rows, table_data):
    if not selected_rows:
        return {}
    row = table_data[selected_rows[0]]
    pairs = [p.strip() for p in row.get('currency_pairs', '').split(',') if p.strip()]
    seq_len = row.get('seq_len', 1)
    return {'pairs': pairs, 'seq_len': seq_len}

# Step B: when the pair list changes, fetch all histories at once
@callback(
    Output('currency-history-store', 'data'),
    Input('currency-pair-store', 'data'),
    prevent_initial_call=True
)
def _fetch_all_histories(store):
    pairs = store.get('pairs', []) if isinstance(store, dict) else []
    out = {}
    for pair in pairs:
        resp = requests.get(f"{API_BASE}/currency-pairs/{pair}/history")
        resp.raise_for_status()
        out[pair] = resp.json()
    return out

# Step C: render one Graph per pair
@callback(
    Output('currency-plots-container', 'children'),
    Input('currency-pair-store', 'data')
)
def _render_graphs(store):
    pairs = store.get('pairs', []) if isinstance(store, dict) else []
    return [
        dcc.Graph(
            id={'type': 'currency-plot', 'index': pair},
            style={'height': '300px'}
        )
        for pair in pairs
    ]

# Step D: this one callback does two things:
#   • On initial mount or when history-store changes → draw base series
#   • On selected-latent-store change → recolor overlays only
@callback(
    Output({'type': 'currency-plot', 'index': MATCH}, 'figure'),
    Input({'type': 'currency-plot', 'index': MATCH}, 'id'),
    Input('currency-history-store', 'data'),
    Input('selected-latent-store', 'data'),
    State('currency-pair-store', 'data'),
    State({'type': 'currency-plot', 'index': MATCH}, 'figure'),
)
def _update_plot(graph_id, all_histories, selected_latents, pair_store, existing_fig):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    pair = graph_id['index']
    seq_len = pair_store.get('seq_len', 1) if isinstance(pair_store, dict) else 1

    print(f'pairs: {pair}, seq_len: {seq_len}')

    # Helper to build base figure from stored history:
    def build_base():
        data = all_histories.get(pair, [])
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['rate'], mode='lines', name=pair))
        return fig, df

    # If we’re here because the user *selected* a SOM cell, just recolor:
    if triggered == 'selected-latent-store' and existing_fig:
        fig = go.Figure(existing_fig)
        # drop any old overlays (keep only the base trace at index 0)
        fig.data = [fig.data[0]]
        df = pd.DataFrame(all_histories.get(pair, []))
        df['date'] = pd.to_datetime(df['date'])

        for sel in selected_latents or []:
            start = pd.to_datetime(sel['start_date'])
            end   = start + pd.Timedelta(days=seq_len - 1)
            mask = (df['date'] >= start) & (df['date'] <= end)
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, 'date'],
                    y=df.loc[mask, 'rate'],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False
                ))
        fig.update_layout(
            title=pair,
            margin={'l':20,'r':20,'t':30,'b':20},
            xaxis_title='Date', yaxis_title='Exchange Rate'
        )
        return fig

    # Otherwise (initial mount or histories just arrived) – build fresh
    fig, df = build_base()
    fig.update_layout(
        title=pair,
        margin={'l':20,'r':20,'t':30,'b':20},
        xaxis_title='Date', yaxis_title='Exchange Rate'
    )
    return fig