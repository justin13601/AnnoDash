#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""

import os
import csv
from datetime import timedelta, datetime as dt
from collections import defaultdict

import dash
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate

import scipy
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
                children="Explore various clinical laboratory measurements. Select a laboratory measurement to "
                         "visualize patient records at different time points. If importing data, please ensure "
                         "columns denoting the label, id, fluid, and category of each measurement is present."
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

# Date
# Format charttime
df_labevents["charttime"] = df_labevents["charttime"].apply(
    lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
)  # String -> Datetime

# define labitem pairs for patient specific tabs
bg_labitem_1 = (50806, 50824)  # Chloride & Sodium, Blood
bg_labitem_2 = (50822, 50808)  # Potassium & Calcium, Blood
bg_labitem_3 = (50821, 50818)  # PO2 & PCO2, Blood

chem_labitem_1 = (50868, 50882)  # Anion Gap & Bicarbonate, Blood
chem_labitem_2 = (50893, 50912)  # Calcium & Creatinine, Blood
chem_labitem_3 = (50902, 50931)  # Chloride & Glucose, Blood
chem_labitem_4 = (50912, 50971)  # Creatinine & Potassium, Blood

cbc_labitem_1 = (51222, 51300)  # Hemoglobin & WBC, Blood

# set labitem pair that will be used
bg_pair = bg_labitem_3
chem_pair = chem_labitem_4
cbc_pair = cbc_labitem_1


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Filter by Category (Optional)"),
            dcc.Dropdown(
                id="category-select",
                value=None,
                style={"border-radius": 0},
                options=[{"label": x, "value": x} for x in list(labitems_dict.keys())],
            ),
            html.Br(),
            html.P("Filter by Fluid (Optional)"),
            dcc.Dropdown(
                id="fluid-select",
                value=None,
                style={"border-radius": 0},
                options=[],
                disabled=True,
            ),
            html.Br(),
            html.P("Select Lab Measurement (Enables Measurement Specific Tabs)"),
            dcc.Dropdown(
                id="labitem-select",
                value=None,
                style={"border-radius": 0},
                options=[{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in
                         labitemsid_dict],
            ),
            html.Br(),
            html.P("Specify Patient (Enables Patient Specific Tabs)"),
            dcc.Dropdown(
                id="patient-select",
                value=None,
                style={"border-radius": 0},
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


def initialize_all_measurements_graph():
    # not sure what to initialize yet
    return


def generate_all_patients_graph(labitem):
    table = df_labevents.query(f'itemid == {labitem}')
    table.replace(np.inf, np.nan)
    table.dropna(subset=['valuenum'], inplace=True)
    if table.empty:
        return {}

    hist_data = [list(table['valuenum'])]
    if hist_data == [[]]:
        return {}
    units = list(table['valueuom'])[0]
    group_labels = [labitemsid_dict[labitem]]
    fig = ff.create_distplot(hist_data, group_labels)
    fig.update_layout(
        title={
            'text': f"Patient Cohort w/ {labitemsid_dict[labitem]}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="verdana",
                size=25,
                color="Black"
            )},
        xaxis_title=f"{labitemsid_dict[labitem]} ({units})",
        yaxis_title="Count (#)",
        font=dict(
            family="verdana",
            size=12,
            color="Black"
        ),
        height=int(600)
    )
    return fig


def generate_violinplot(labitem):
    table = df_labevents.query(f'itemid == {labitem}')
    table.replace(np.inf, np.nan)
    table.dropna(subset=['valuenum'], inplace=True)
    if table.empty:
        return {}

    units = list(table['valueuom'])[0]
    fig = go.Figure(data=go.Violin(y=table['valuenum'], box_visible=True, meanline_visible=True, points='all',
                                   x0=f"Lab Item ID: {labitem}"))

    fig.update_layout(
        yaxis_title=f"{labitemsid_dict[labitem]} ({units})",
        font=dict(
            family="verdana",
            size=12,
            color="Black"
        ),
        margin=dict(t=10, b=10),
        height=int(400)
    )
    return fig


def generate_tab_graph(labitem, patient, template_labitems):
    table_labitem_1 = df_labevents.query(f'itemid == {template_labitems[0]}')
    table_labitem_2 = df_labevents.query(f'itemid == {template_labitems[1]}')
    table_labitem_target = df_labevents.query(f'itemid == {labitem}')
    table_labitem_patient_1 = table_labitem_1.query(f'subject_id == {patient}')
    table_labitem_patient_2 = table_labitem_2.query(f'subject_id == {patient}')
    table_labitem_patient_target = table_labitem_target.query(f'subject_id == {patient}')

    if table_labitem_patient_1.empty or table_labitem_patient_2.empty or table_labitem_patient_target.empty:
        return {}

    start_date = min(min(table_labitem_patient_target['charttime']),
                     min(table_labitem_patient_target['charttime'])) - timedelta(hours=12)
    end_date = start_date + timedelta(hours=96) + timedelta(hours=12)

    mask_1 = (table_labitem_patient_1['charttime'] > start_date) & (table_labitem_patient_1['charttime'] <= end_date)
    table_labitem_patient_1 = table_labitem_patient_1.loc[mask_1]

    mask_2 = (table_labitem_patient_2['charttime'] > start_date) & (table_labitem_patient_2['charttime'] <= end_date)
    table_labitem_patient_2 = table_labitem_patient_2.loc[mask_2]

    mask_plot = (table_labitem_patient_target['charttime'] > start_date) & (
            table_labitem_patient_target['charttime'] <= end_date)
    table_labitem_patient_target = table_labitem_patient_target.loc[mask_plot]

    series_names = [labitemsid_dict[template_labitems[0]], labitemsid_dict[template_labitems[1]],
                    labitemsid_dict[labitem]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=table_labitem_patient_1["charttime"], y=table_labitem_patient_1["valuenum"], name=series_names[0]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_labitem_patient_2["charttime"], y=table_labitem_patient_2["valuenum"], name=series_names[1]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_labitem_patient_target["charttime"], y=table_labitem_patient_target["valuenum"],
                   name=series_names[2]),
        secondary_y=True,
    )

    fig.update_traces(mode="markers+lines", hovertemplate='Value: %{y:.1f}<extra></extra>')

    fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker')
    fig.update_yaxes(title_text=f"Value ({list(table_labitem_patient_1['valueuom'])[0]})", showspikes=True,
                     spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker', secondary_y=False)
    fig.update_yaxes(title_text=f"Value ({list(table_labitem_target['valueuom'])[0]})", showspikes=True,
                     spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker', secondary_y=True)

    fig.update_layout(
        title={
            'text': f"{labitemsid_dict[labitem]} vs. {labitemsid_dict[template_labitems[0]]} and {labitemsid_dict[template_labitems[1]]}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="verdana",
                size=20,
                color="Black"
            )},
        xaxis_title="Time (Hours)",
        font=dict(
            family="verdana",
            size=12,
            color="Black"
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100),
        height=int(600)
    )
    return fig


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
                    style={'height': '500%'},
                    children=[
                        html.H4("Patient Records"),
                        html.Hr(style={}),
                        html.Br(),
                        dcc.Tabs([
                            dcc.Tab(label='All Measurements', id="all_measurements_tab",
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '600px'},
                                            id="all_measurements_graph",
                                            figure={}
                                        )
                                    ]),
                            dcc.Tab(label='All Patients', id="all_patients_tab", disabled=True,
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '600px'},
                                            id="all_patients_graph",
                                            figure={}
                                        )
                                    ]),
                            dcc.Tab(label='Blood Gas', id="blood_gas_tab", disabled=True,
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '600px'},
                                            id="blood_gas_graph",
                                            figure={}
                                        )
                                    ]),
                            dcc.Tab(label='Chemistry', id="chemistry_tab", disabled=True,
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '600px'},
                                            id="chemistry_graph",
                                            figure={}
                                        )
                                    ]),
                            dcc.Tab(label='Complete Blood Count', id="cbc_tab", disabled=True,
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '600px'},
                                            id="cbc_graph",
                                            figure={}
                                        )
                                    ]),
                        ])
                    ],
                ),
                # Patient violinplot summaries by lab measurements
                html.Div(
                    id="violinplot_card",
                    style={"display": "none"},
                    children=[
                        html.Br(),
                        html.Hr(),
                        html.Br(),
                        dcc.Graph(style={'height': '400px'}, id="patient_violinplot", figure={}),
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
        Input("labitem-select", "value"),
    ],
)
def reset_values(n_clicks, category, labitem):
    if n_clicks == 0 and category is None:
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
    Output("all_patients_tab", "disabled"),
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

        table1 = df_labevents.query(f'itemid == {bg_pair[0]}')
        table2 = df_labevents.query(f'itemid == {bg_pair[1]}')

        table3 = df_labevents.query(f'itemid == {chem_pair[0]}')
        table4 = df_labevents.query(f'itemid == {chem_pair[1]}')

        table5 = df_labevents.query(f'itemid == {cbc_pair[0]}')
        table6 = df_labevents.query(f'itemid == {cbc_pair[1]}')

        patients_with_labitem = set(table['subject_id'].unique())
        patients_with_pair_item1 = set(table1['subject_id'].unique())
        patients_with_pair_item2 = set(table2['subject_id'].unique())
        patients_with_pair_item3 = set(table3['subject_id'].unique())
        patients_with_pair_item4 = set(table4['subject_id'].unique())
        patients_with_pair_item5 = set(table5['subject_id'].unique())
        patients_with_pair_item6 = set(table6['subject_id'].unique())

        temp_set_1 = patients_with_labitem.intersection(patients_with_pair_item1)
        temp_set_2 = temp_set_1.intersection(patients_with_pair_item2)
        temp_set_3 = temp_set_2.intersection(patients_with_pair_item3)
        temp_set_4 = temp_set_3.intersection(patients_with_pair_item4)
        temp_set_5 = temp_set_4.intersection(patients_with_pair_item5)
        patient_list = list(temp_set_5.intersection(patients_with_pair_item6))
        patient_list.sort()

        options = [{"label": each_patient, "value": each_patient} for each_patient in
                   patient_list]
        disabled = False
        return options, disabled, disabled
    return options, disabled, disabled


@app.callback(
    Output("all_patients_graph", "figure"),
    Output("blood_gas_graph", "figure"),
    Output("blood_gas_tab", "disabled"),
    Output("chemistry_graph", "figure"),
    Output("chemistry_tab", "disabled"),
    Output("cbc_graph", "figure"),
    Output("cbc_tab", "disabled"),
    [
        Input("labitem-select", "value"),
        Input("patient-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_graph(labitem, patient, reset):
    disabled = True

    if labitem is None:
        return {}, {}, disabled, {}, disabled, {}, disabled

    if patient:
        disabled = False
        return generate_all_patients_graph(labitem), \
               generate_tab_graph(labitem, patient, bg_pair), disabled, \
               generate_tab_graph(labitem, patient, chem_pair), disabled, \
               generate_tab_graph(labitem, patient, cbc_pair), disabled

    return generate_all_patients_graph(labitem), {}, disabled, {}, disabled, {}, disabled


@app.callback(
    Output("violinplot_card", "style"),
    Output("patient_violinplot", "figure"),
    [
        Input("labitem-select", "value"),
        # Input("all_patient_graph", "clickData")     # maybe show some info about flags (abnormal) in violinplot card when hover/clicked on datapoint?
    ]
)
def update_violinplot(labitem):
    visible = {"display": "block"}
    hidden = {"display": "none"}
    if labitem is None:
        return hidden, {}
    return visible, generate_violinplot(labitem)


# run
if __name__ == "__main__":
    app.run_server(debug=True)
