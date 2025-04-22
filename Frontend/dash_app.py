# Frontend/dash_app.py
import dash
from dash import html, dcc, Input, Output, State, dash_table
import requests

# Import components
from components.selector import SelectorSection
from components.modals import ConfigModals

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose Flask server

# Layout
app.layout = html.Div([
    html.H1("ScenGen Config Dashboard"),
    SelectorSection(),
    ConfigModals(),
])

if __name__ == '__main__':
    # Use app.run instead of run_server per Dash v2.14+
    app.run(debug=True)
