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

df_loinc = load_data(os.path.join(PATH_data, 'LoincTableCoreCondensed.csv'))
print("LOINC codes loaded.\n")

labitemsid_dict = pd.Series(df_labitems.label.values, index=df_labitems.itemid.values).to_dict()

loinc_dict = pd.Series(df_loinc.COMPONENT.values, index=df_loinc.LOINC_NUM.values).to_dict()

df_labitems_annotation = pd.DataFrame()
unannoted = labitemsid_dict
if os.path.isfile('annotations.csv'):
    try:
        df_labitems_annotation = pd.read_csv('annotations.csv')
        df_unannoted = df_labitems_annotation[df_labitems_annotation['Annotation_Code'].isnull()]
        unannoted = pd.Series(df_unannoted.label.values, index=df_unannoted.itemid.values).to_dict()
    except pd.errors.EmptyDataError:
        df_labitems_annotation = df_labitems
        df_labitems_annotation[["Annotation_Label", "Annotation_Code"]] = None
else:
    df_labitems_annotation = df_labitems
    df_labitems_annotation[["Annotation_Label", "Annotation_Code"]] = None

# Date
# Format charttime
df_labevents["charttime"] = df_labevents["charttime"].apply(
    lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
)  # String -> Datetime

# define labitem pairs for patient specific tabs
bg_pair = (50821, 50818)  # PO2 & PCO2, Blood
chem_pair = (50912, 50971)  # Creatinine & Potassium, Blood
cbc_pair = (51222, 51300)  # Hemoglobin & WBC, Blood

first_value_testing = 0  ############################## FOR TESTING ##############################


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
                children=""
            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Lab Measurement"),
            dcc.Dropdown(
                id="labitem-select",
                value=initialize_labitem_select()[1],
                style={"border-radius": 0},
                options=initialize_labitem_select()[0],
            ),
            html.Br(),
            html.P("Specify Patient (Enables Patient Specific Tabs)"),
            dcc.Dropdown(
                id="patient-select",
                value=initialize_patient_select()[1],
                style={"border-radius": 0},
                options=initialize_patient_select()[0],
                disabled=False,
            ),
            html.Br(),
            html.Hr(),
            html.Br(),
            html.P("Annotate"),
            # dcc.Dropdown(
            #     id="annotate-select",
            #     value=None,
            #     placeholder='Start typing...',
            #     style={"border-radius": 0},
            #     options=initialize_annotate_select(),
            # ),
            html.Div(
                children=dcc.Input(
                    id="annotate-text",
                    placeholder="Annotation...",
                    debounce=True,
                    style={"width": 390},
                    autoFocus=True,
                ),
            ),
            html.Br(),
            html.Div(
                id="submit-btn-outer",
                children=html.Button(id="submit-btn", children="Submit & Next", n_clicks=0,
                                     style={'width': '100%', 'color': 'white'}),
            ),
        ],
        style={'width': '100%', 'color': 'black'},
    )


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
    group_labels = [f"{labitemsid_dict[labitem]} (%)"]
    fig = ff.create_distplot(hist_data, group_labels, colors=['rgb(44,140,255)'])
    fig.update_layout(
        title={
            'text': labitemsid_dict[labitem],
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="verdana",
                size=25,
                color="Black"
            )},
        xaxis_title=f"{labitemsid_dict[labitem]} ({units})",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(
            family="verdana",
            size=12,
            color="Black"
        ),
        height=int(502),
        margin=dict(l=50, r=50, t=90, b=20),
    )
    return fig


def generate_tab_graph(labitem, patient, template_labitems):
    table_labitem_1 = df_labevents.query(f'itemid == {template_labitems[0]}')
    table_labitem_2 = df_labevents.query(f'itemid == {template_labitems[1]}')
    table_labitem_target = df_labevents.query(f'itemid == {labitem}')
    table_labitem_patient_1 = table_labitem_1.query(f'subject_id == {patient}')
    table_labitem_patient_2 = table_labitem_2.query(f'subject_id == {patient}')
    table_labitem_patient_target = table_labitem_target.query(f'subject_id == {patient}')
    units_1 = list(table_labitem_1['valueuom'])[0]
    units_2 = list(table_labitem_2['valueuom'])[0]
    units_target = list(table_labitem_target['valueuom'])[0]

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
        go.Scatter(x=table_labitem_patient_1["charttime"], y=table_labitem_patient_1["valuenum"],
                   name=f"{series_names[0]} ({units_1})"), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_labitem_patient_2["charttime"], y=table_labitem_patient_2["valuenum"],
                   name=f"{series_names[1]} ({units_2})"), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_labitem_patient_target["charttime"], y=table_labitem_patient_target["valuenum"],
                   name=f"{series_names[2]} ({units_target})"),
        secondary_y=True,
    )

    fig.update_traces(mode="markers+lines", hovertemplate='Value: %{y:.1f}<extra></extra>')

    fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker')
    fig.update_yaxes(showspikes=True,
                     spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker', secondary_y=False)
    fig.update_yaxes(showspikes=True,
                     spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                     spikemode='across+marker', secondary_y=True)

    fig.update_layout(
        title={
            'text': f"{labitemsid_dict[labitem]}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="verdana",
                size=25,
                color="Black"
            )},
        xaxis_title="Time (Hours)",
        font=dict(
            family="verdana",
            size=12,
            color="Black"
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95),
        margin=dict(l=50, r=0, t=90, b=20),
        height=int(502)
    )
    return fig


def query_patients(labitem):
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

    temp_set_1 = patients_with_pair_item1.intersection(patients_with_pair_item2)
    temp_set_2 = temp_set_1.intersection(patients_with_pair_item3)
    temp_set_3 = temp_set_2.intersection(patients_with_pair_item4)
    temp_set_4 = temp_set_3.intersection(patients_with_pair_item5)
    temp_set_5 = temp_set_4.intersection(patients_with_pair_item6)
    temp_set_6 = temp_set_5.intersection(patients_with_labitem)

    patient_list = list(temp_set_6)
    patient_list.sort()

    patients = [{"label": each_patient, "value": each_patient} for each_patient in
                patient_list]
    return patients


def initialize_all_patients_graph():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    fig = generate_all_patients_graph(labitems[first_value_testing]["value"])
    return fig


def initialize_tab_graph(pair):
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    patients = query_patients(labitems[first_value_testing]["value"])
    if not patients[0]:
        return {}
    first_patient = patients[0]["value"]
    fig = generate_tab_graph(labitems[first_value_testing]["value"], first_patient, template_labitems=pair)
    return fig


def initialize_tab():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    patients = query_patients(labitems[first_value_testing]["value"])
    if not patients[0]:
        return True
    first_patient = patients[0]["value"]
    disabled = update_graph(labitems[first_value_testing]["value"], first_patient, submit=0)[2]
    return disabled


def initialize_labitem_select():
    options = [{"label": f'{each_id}: {unannoted[each_id]}', "value": each_id} for each_id in unannoted]
    first_labitem = options[first_value_testing]["value"]
    return options, first_labitem


def initialize_patient_select():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    options = query_patients(labitems[first_value_testing]["value"])
    if not options:
        return [], None
    first_labitem = options[0]["value"]
    return options, first_labitem


def initialize_annotate_select():
    # loinc_codes = [{"label": f'{each_code}: {loinc_dict[each_code]}', "value": each_code} for each_code in loinc_dict]
    # return loinc_codes
    return []


def annotate(labitem, annotation):
    df_labitems_annotation.loc[(df_labitems_annotation.itemid == labitem), ['Annotation_Label', 'Annotation_Code']] = [
        annotation, annotation]
    # df_labitems_annotation.loc[(df_labitems_annotation.itemid == labitem), ['Annotation_Label', 'Annotation_Code']] = [
    #     annotation, loinc_dict[annotation]]
    #use json per labitem
    df_labitems_annotation.to_csv("annotations.csv", index=False)
    return


def update_labitem_options(options, labitem):
    next_value = options[0]['value']
    for index in range(len(options)):
        if options[index]['value'] == labitem:
            del options[index]
            next_value = options[index]['value']
            break
    return options, next_value


######################################################################################################
@app.callback(
    Output("submit-btn", "n_clicks"),
    # Output('annotate-select', 'value'),
    Output('annotate-text', 'value'),
    [
        Input("submit-btn", "n_clicks"),
    ],
    [
        State('labitem-select', 'value'),
        # State('annotate-select', 'value'),
        State('annotate-text', 'value')
    ]
)
def submit_annotation(n_clicks, labitem, annotation):
    if n_clicks == 0:
        raise PreventUpdate
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if n_clicks > 0 and triggered_id == 'submit-btn.n_clicks':
        annotate(labitem, annotation)
        n_clicks = 0
        return n_clicks, ''


@app.callback(
    Output("tabs", "value"),
    [
        Input("patient-select", "value"),
    ]
)
def update_tabs_view(patient):
    if patient is not None:
        raise PreventUpdate
    else:
        return "home-tab"


@app.callback(
    Output("labitem-select", "options"),
    Output("labitem-select", "value"),
    [
        Input("submit-btn", "n_clicks"),
    ],
    [
        State('labitem-select', 'value'),
        State('labitem-select', 'options'),
    ]
)
def update_lab_measurement_dropdown(submit, value, options):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    curr_options = options
    new_value = curr_options[0]["value"]
    if triggered_id == 'submit-btn.n_clicks':
        new_options, new_value = update_labitem_options(curr_options, value)
        curr_options = new_options
    return curr_options, new_value


@app.callback(
    Output("patient-select", "options"),
    Output("patient-select", "disabled"),
    Output("patient-select", "value"),
    [
        Input("labitem-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_patient_dropdown(labitem, submit):
    options = []
    disabled = True
    if labitem:
        options = query_patients(labitem)
        disabled = False
        if options:
            first_patient = options[0]["value"]
            return options, disabled, first_patient
        return options, disabled, None
    return options, disabled, None


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
        Input("submit-btn", "n_clicks"),
    ],
)
def update_graph(labitem, patient, submit):
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
                            dcc.Tab(label='All Patients', id="all_patients_tab", disabled=False,
                                    value='home-tab',
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '502px'},
                                            id="all_patients_graph",
                                            figure=initialize_all_patients_graph()
                                        )
                                    ]),
                            dcc.Tab(label='Blood Gas', id="blood_gas_tab", disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '502px'},
                                            id="blood_gas_graph",
                                            figure=initialize_tab_graph(bg_pair)
                                        )
                                    ]),
                            dcc.Tab(label='Chemistry', id="chemistry_tab", disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '502px'},
                                            id="chemistry_graph",
                                            figure=initialize_tab_graph(chem_pair)
                                        )
                                    ]),
                            dcc.Tab(label='Complete Blood Count', id="cbc_tab", disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '502px'},
                                            id="cbc_graph",
                                            figure=initialize_tab_graph(cbc_pair)
                                        )
                                    ]),
                        ], id='tabs', value='home-tab')
                    ],
                ),
            ],
        ),
    ],
)
######################################################################################################


# run
if __name__ == "__main__":
    app.run_server(port=8888, debug=True)
