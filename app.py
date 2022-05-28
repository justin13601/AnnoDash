#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""

import os
import csv
import datetime
from datetime import datetime as dt
from collections import defaultdict

import dash
import plotly.express as px
from dash import Dash, html, dcc, dash_table
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
from google.cloud import bigquery


def big_query(query):
    client = bigquery.Client()
    query_job = client.query(query)  # API request
    print("The query data:")
    for row in query_job:
        # row values can be accessed by field name or index
        print("name={}, count={}".format(row[0], row["total_people"]))
    return


######################################################################################################
app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)

app.title = "MIMIC-IV Clinical Laboratory Data Dashboard"
server = app.server
app.config.suppress_callback_exceptions = True


######################################################################################################

# callback_manager.attach_to_app(app)


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("MIMIC-IV Clinical Dashboard"),
            html.H3("Welcome to MIMIC-IV Clinical Laboratory Data"),
            html.P(
                id="intro",
                children="Explore various clinical laboratory measurements. Click\
                 on the heatmap to visualize patient chart measurements at different time points. If importing data, "
                         "please ensure a dictionary mapping each label with a code is present.",
            ),
        ],
    )


######################################################################################################
# path
PATH_base = os.getcwd()
PATH_data = os.path.join(PATH_base, "demo-data")


# load data
def load_data(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    df_data = pd.read_csv(path)
    print('Done.\n')
    return df_data


df_labitems = load_data(os.path.join(PATH_data, 'D_LABITEMS.csv'))
df_labevents = load_data(os.path.join(PATH_data, 'LABEVENTS.csv'))
print("Data loaded.\n")

df_labitems['category'] = [x.lower().capitalize() for x in df_labitems['category'].tolist()]
df_labitems['fluid'] = [x.lower().capitalize() for x in df_labitems['fluid'].tolist()]

labitems_tuples = list(df_labitems[['category', 'fluid', 'itemid', 'label']].itertuples(index=False))
labitems_dict = defaultdict(lambda: defaultdict(dict))
for i in range(len(labitems_tuples)):  # Each tuple is "key1, key2, value"
    labitems_dict[labitems_tuples[i][0]][labitems_tuples[i][1]][labitems_tuples[i][2]] = labitems_tuples[i][3]

labitemsid_dict = pd.Series(df_labitems.label.values, index=df_labitems.itemid.values).to_dict()

categories_list = df_labitems["category"].unique().tolist()
fluids_list = df_labitems["fluid"].unique().tolist()
patients_list = df_labevents["subject_id"].unique().tolist()

# Date
# Format charttime
df_labevents["charttime"] = df_labevents["charttime"].apply(
    lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
)  # String -> Datetime


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Filter by Category"),
            dcc.Dropdown(
                id="category-select",
                value=None,
                options=[{"label": x, "value": x} for x in list(labitems_dict.keys())],
            ),
            html.Br(),
            html.P("Filter by Fluid"),
            dcc.Dropdown(
                id="fluid-select",
                value=None,
                options=[],
                disabled=True,
            ),
            html.Br(),
            html.P("Select Lab Measurement"),
            dcc.Dropdown(
                id="labitem-select",
                value=None,
                options=[{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in
                         labitemsid_dict],
            ),
            html.Br(),
            html.P("Specify Patient"),
            dcc.Dropdown(
                id="patient-select",
                value=None,
                options=[],
                disabled=True,
            ),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0,
                                     style={'width': '100%', 'color': 'white'}),
            ),
        ],
        style={'width': '100%', 'color': 'black'},
    )


def initialize_all_patients_graph():
    # clustering of all labitems values?
    return


def initialize_tab_graph():
    # plot 2 values per tab in tabs 2-4
    return


def initialize_boxplot():
    # not sure what to initialize yet
    return


def generate_all_patients_graph():
    # histogram
    return


def generate_tab_graph():
    # overlay 2 values with labitem
    return


def generate_boxplot():
    # boxplot dist
    return


######################################################################################################
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("mimic.png"), style={'height': '120%', 'width': '10%'})],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()] + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Tabbed graphs
                html.Div(
                    id="patient_card",
                    children=[
                        html.H4("Patient Records"),
                        html.Hr(style={}),
                        html.Br(),
                        dcc.Tabs([
                            dcc.Tab(label='All Patients', children=[
                                dcc.Graph(
                                    id="all_patients_graph",
                                    figure={}
                                )
                            ]),
                            dcc.Tab(label='Blood Gas', children=[
                                dcc.Graph(
                                    id="blood_gas_graph",
                                    figure={}
                                )
                            ]),
                            dcc.Tab(label='Chemistry', children=[
                                dcc.Graph(
                                    id="chemistry_graph",
                                    figure={}
                                )
                            ]),
                            dcc.Tab(label='Vital Signs', children=[
                                dcc.Graph(
                                    id="vital_sign_graph",
                                    figure={}
                                )
                            ]),
                        ])
                    ],
                ),
                # Patient boxplot summaries by lab measurements
                html.Div(
                    id="boxplot_card",
                    children=[
                        html.Br(),
                        html.Hr(),
                        html.Br(),
                        dcc.Graph(id="patient_boxplot", figure={}),
                    ],
                ),
            ],
        ),
    ],
)


######################################################################################################
@app.callback(
    Output("category-select", "value"),
    Output("fluid-select", "value"),
    Output("labitem-select", "value"),
    Output("patient-select", "value"),
    Output("reset-btn", "n_clicks"),
    [
        Input("reset-btn", "n_clicks"),
        Input("category-select", "value"),
    ],
)
def reset_values(n_clicks, category):
    if n_clicks is None or category is None:
        raise PreventUpdate
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'category-select.value':
        return category, None, None, None, n_clicks
    if n_clicks > 0 and triggered_id == 'reset-btn.n_clicks':
        n_clicks = 0
        return None, None, None, None, n_clicks


@app.callback(
    Output("fluid-select", "options"),
    Output("fluid-select", "disabled"),
    [
        Input("category-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_fluid_dropdown(category, reset):
    options = []
    disabled = True
    if category:
        options = [{"label": each_fluid, "value": each_fluid} for each_fluid in list(labitems_dict[category].keys())]
        disabled = False
        return options, disabled
    return options, disabled


@app.callback(
    Output("labitem-select", "options"),
    [
        Input("category-select", "value"),
        Input("fluid-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_lab_measurement_dropdown(category, fluid, reset):
    options = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    if category:
        options = []
        for each in labitems_dict['Blood gas']:
            options.append(labitems_dict[category][each])
        measurements = {k: v for d in options for k, v in d.items()}
        options = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in measurements]
        if fluid:
            options = [{"label": f'{each_measurement}: {labitems_dict[category][fluid][each_measurement]}',
                        "value": each_measurement} for each_measurement in
                       labitems_dict[category][fluid]]
    return options


@app.callback(
    Output("patient-select", "options"),
    Output("patient-select", "disabled"),
    [
        Input("labitem-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_patient_dropdown(labitem, reset):
    options = []
    disabled = True
    if labitem:
        table = df_labevents.query(f'itemid == {labitem}')
        options = [{"label": each_patient, "value": each_patient} for each_patient in
                   list(table['subject_id'].unique())]
        disabled = False
        return options, disabled
    return options, disabled


@app.callback(
    Output("all_patients_graph", "figure"),
    Output("blood_gas_graph", "figure"),
    Output("chemistry_graph", "figure"),
    Output("vital_sign_graph", "figure"),
    Output("patient_boxplot", "figure"),
    [
        Input("labitem-select", "value"),
        Input("patient-select", "value"),
    ],
)
def update_graph(labitem, patient):
    # updates patient graph after changing dropdowns
    return


# @app.callback(
#     Output("patient_boxplot", "figure"), #boxplot card instead of boxplot figure
#     [
#         Input("labitem-select", "value"),
#         Input("all_patient_graph", "clickData"),
#     ]
# )
# def update_boxplot():
#     # maybe show some info about flags (abnormal) in boxplot card when hover/clicked on datapoint?
#     return


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("patient_boxplot", "figure")],
)

# run
if __name__ == "__main__":
    app.run_server(debug=True)
