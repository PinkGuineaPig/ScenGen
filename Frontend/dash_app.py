# Frontend/dash_app.py

from dash import Dash, html
from Frontend.components.selector import ConfigTableSection  # import the table-only component
from Frontend.components.modals import ConfigModals
from Frontend.components.plots import ConfigPlots   # NEW

# Initialize Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

# Layout
app.layout = html.Div([
    ConfigTableSection(),  # use the table-only section
    ConfigModals(),
    ConfigPlots()
])

# Expose main entrypoint for console-script
def main():
    # Use the new run() method
    app.run(host="0.0.0.0", port=8050)

# Allow direct python invocation
if __name__ == '__main__':
    main()