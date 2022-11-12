from dash import Dash, html, dcc, dash_table, ALL, ctx
from dash.dependencies import State, Input, Output, ClientsideFunction
from src.app.app import app
from src.app.layouts import serve_layout
import src.app.callbacks

server = app.server

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.Div(
            id='dashboard-content',
            children=[],
        )
    ]
)


@app.callback(
    Output('dashboard-content', 'children'),
    [
        Input('url', 'href'),
    ],
)
def display_page(href):
    try:
        return serve_layout()
    except:
        print('Something went wrong!')
        return '404'


# run main.py (MIMIC-Dash v2)
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
