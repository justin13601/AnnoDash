import dash
from dash import Dash, html, dcc, dash_table
import pandas as pd
from dash.dependencies import Input, Output

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

df = df.reset_index().rename(columns={"index": "id"})

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H4('Multiple page datatables - active cell information'),

    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns if i != 'id'],
        data=df.to_dict('records'),
        page_size=5),

    dcc.Markdown(id='test_cell'),
])


@app.callback(
    Output('test_cell', 'children'),
    Input('table', 'active_cell'))
def return_cell_info(active_cell):
    return str(active_cell)


if __name__ == '__main__':
    app.run_server(debug=True)
