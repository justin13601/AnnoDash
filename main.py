from dash import Dash, html, dcc, dash_table, ALL, ctx
from dash.dependencies import State, Input, Output, ClientsideFunction
from app import app
from layouts import main_layout
import callbacks

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
        Input('url', 'pathname'),
    ],
)
def display_page(pathname):
    try:
        return main_layout
    except:
        print('Something went wrong!')
        return '404'


# run app.py (MIMIC-Dash v2)
if __name__ == "__main__":
    # app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
    app.run_server(port=8888, debug=True, use_reloader=False)
