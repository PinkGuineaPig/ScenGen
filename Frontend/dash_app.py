# Frontend/dash_app.py

from dash import Dash, html
import dash

from Frontend.components.selector import ConfigTableSection
from Frontend.components.modals    import ConfigModals
from Frontend.components.plots     import ConfigPlots
from Frontend.components.currency_plots import CurrencyPlots

# Initialize Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

app.layout = html.Div([

    html.Details(
        className='chapter',
        open=True,
        children=[
            html.Summary("1. Data Selection"),
            html.Div(ConfigTableSection())
        ]
    ),

    # 2) Train Models
    html.Details(
        className='chapter',
        open=True,
        children=[
            html.Summary("2. Train Models"),
            html.Div(ConfigModals())
        ]
    ),

    # 3) Analysis: SOM & PCA
    html.Details(
        className='chapter',
        open=False,
        children=[
            html.Summary("3. Analysis: SOM & PCA"),
            html.Div(ConfigPlots()),
            html.Div(CurrencyPlots())
        ]
    )
])


def main():
    app.run(host="0.0.0.0", port=8050)

if __name__ == '__main__':
    main()
