#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""

import os
import re
import errno
import csv
import time
import json
import yaml
import jaro
import pickle
from datetime import timedelta, datetime as dt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from ftfy import fix_text
from collections import defaultdict

import dash
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate
from flask import Flask, send_file
from zipfile import ZipFile

import scipy
import numpy as np
import pandas as pd
from google.cloud import bigquery

from tf_idf_matrix import cosine_similarity, TfidfVectorizer, ngrams


# from callbacks.all_callbacks import callback_manager


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

app.title = "Clinical Laboratory Data Dashboard"
server = app.server
app.config.suppress_callback_exceptions = True


######################################################################################################
# callback_manager.attach_to_app(app)


# load data
def load_data(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    if path == os.path.join(PATH_data, 'LoincTableCore.csv'):
        df_data = pd.read_csv(path, dtype=object)
    else:
        df_data = pd.read_csv(path)
    print('Done.\n')
    return df_data


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.safe_load(f)
        print('Done.\n')
        return configurations


def load_annotations(path):
    annotation_files = [each_json for each_json in os.listdir(path) if each_json.endswith('.json')]
    annotated = [int(each_labitem.strip('.json')) for each_labitem in annotation_files]
    return annotated


def download_annotation(annotation):
    results_folder = 'results-json'
    path = os.path.join(results_folder, f"{annotation}.json")
    return send_file(path, as_attachment=True)


config_file = 'config.yaml'
if os.path.exists(config_file):
    print('Configuration file found.')
    config = load_config('config.yaml')
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

# path
PATH_data = config['directories']['data']
PATH_results = config['directories']['results']
PATH_loinc = config['loinc']['location']

df_labitems = load_data(os.path.join(PATH_data, 'D_LABITEMS.csv'))
df_labevents = load_data(os.path.join(PATH_data, 'LABEVENTS.csv'))
print("Data loaded.\n")

df_loinc = load_data(os.path.join(PATH_loinc, 'LoincTableCore.csv'))
df_loinc = df_loinc[df_loinc['CLASSTYPE'] == str(config['loinc']['loinc-class-type-value'])]
df_loinc.drop(df_loinc[df_loinc.STATUS != 'ACTIVE'].index, inplace=True)
df_loinc.drop(['CLASSTYPE', 'STATUS', 'EXTERNAL_COPYRIGHT_NOTICE', 'VersionFirstReleased', 'VersionLastChanged'],
              axis=1,
              inplace=True)
print(f"LOINC codes (CLASSTYPE={config['loinc']['loinc-class-type-value']}, "
      f"{config['loinc']['loinc-class-type-label']}) loaded and processed.\n")

labitemsid_dict = pd.Series(df_labitems.label.values, index=df_labitems.itemid.values).to_dict()

loinc_dict = pd.Series(df_loinc.LONG_COMMON_NAME.values, index=df_loinc.LOINC_NUM.values).to_dict()
df_loinc_new = pd.DataFrame(
    {'LOINC_NUM': list(loinc_dict.keys()), 'LONG_COMMON_NAME': list(loinc_dict.values())})
df_loinc_new = df_loinc_new.reset_index().rename(columns={"index": "id"})

# load tf-idf matrix if chosen as scorer:
if config['loinc']['related']['scorer'] == 'tf-idf':
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, vocabulary=pickle.load(open(os.path.join(PATH_data, 'LOINC_vectorizer_vocabulary_n=10.pkl'), "rb")))
    vectorizer.fit_transform(list(df_loinc_new['LONG_COMMON_NAME'].unique()))
    tf_idf_matrix = pickle.load(open(os.path.join(PATH_data, 'LOINC_tf_idf_matrix_n=10.pkl'), "rb"))

annotated_list = load_annotations(PATH_results)
unannotated_list = list(set(labitemsid_dict.keys()) - set(annotated_list))
unannotated_list.sort()

# Date
# Format charttime
df_labevents["charttime"] = df_labevents["charttime"].apply(
    lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
)  # String -> Datetime

# define labitem pairs for patient specific tabs
pairs = []
for each in config['graphs']['pairs'].values():
    pairs.append((each['item_1'], each['item_2']))
bg_pair = pairs[0]  # Default PO2 & PCO2, Blood       # Could add FiO2
chem_pair = pairs[1]  # Default Creatinine & Potassium, Blood         # Could also use Sodium & Glucose (overlay 4?)
cbc_pair = pairs[2]  # Default Hemoglobin & WBC, Blood        # Could add RBC (the one with space) -> no observations :(


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Welcome to MIMIC Clinical Dashboard!"),
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
            html.Div(
                id="upload-outer",
                hidden=initialize_upload_field(config),  # if config present, hide, else show upload option
                children=[
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.P('Drag and Drop or Select Configuration File'),
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'borderColor': 'gray'
                        },
                    ),
                    html.Br(),
                ]
            ),
            html.P("Select Lab Measurement:"),
            dcc.Dropdown(
                id="labitem-select",
                value=initialize_labitem_select()[1],
                style={"border-radius": 0},
                options=initialize_labitem_select()[0],
            ),
            html.Br(),
            html.P("Specify Patient (enables patient specific tabs):"),
            dcc.Dropdown(
                id="patient-select",
                value=initialize_patient_select()[1],
                style={"border-radius": 0},
                options=initialize_patient_select()[0],
                disabled=False,
            ),
            html.Br(),
            html.Hr(),
            html.P("Annotate:"),
            dcc.Dropdown(
                id="annotate-select",
                value=None,
                optionHeight=65,
                placeholder='Start typing...',
                style={"border-radius": 0},
                options=initialize_annotate_select(),
            ),
            # html.Div(
            #     children=dcc.Input(
            #         id="annotate-text",
            #         placeholder="Annotation...",
            #         debounce=True,
            #         style={"width": 390},
            #         autoFocus=True,
            #     ),
            # ),
            # html.Br(),
            # html.Div(
            #     id="search-btn-outer",
            #     children=html.Button(id="search-btn", children="Search", n_clicks=0,
            #                          style={'width': '100%', 'color': 'white'}),
            # ),
            html.Div(
                id="annotation-outer",
                hidden=True,
                children=[
                    html.Br(),
                    html.Br(id='before_loading', hidden=False),
                    dcc.Loading(
                        id="related-loading",
                        type="dot",
                        color='#2c89f2',
                        children=[
                            html.P(children=[
                                html.B('Related Results (click on rows for more info):'),
                            ]),
                            html.Div(
                                id="related-datatable-outer",
                                className='related-datable',
                                hidden=False,
                                children=dash_table.DataTable(id='related-datatable',
                                                              data=None,
                                                              columns=[
                                                                  {'name': "LOINC_NUM",
                                                                   'id': 'LOINC_NUM'},
                                                                  {'name': "LONG_COMMON_NAME",
                                                                   'id': 'LONG_COMMON_NAME'}
                                                              ],
                                                              style_data={
                                                                  'whiteSpace': 'normal',
                                                                  'height': 'auto',
                                                                  'lineHeight': '15px',
                                                              },
                                                              style_table={
                                                                  'height': '150px',
                                                                  'overflowY': 'auto'
                                                              },
                                                              style_cell={
                                                                  'textAlign': 'left',
                                                                  'backgroundColor': 'transparent'
                                                              },
                                                              style_header={
                                                                  'fontWeight': 'bold',
                                                                  'color': '#2c8cff'
                                                              },
                                                              style_data_conditional=[
                                                                  {  # 'active' | 'selected'
                                                                      'if': {'state': 'active'},
                                                                      'backgroundColor': 'transparent',
                                                                      'border': '1px solid lightgray'
                                                                  },
                                                                  {
                                                                      'if': {'column_id': 'LOINC_NUM'},
                                                                      'width': '30%'
                                                                  },
                                                              ],
                                                              page_size=10,
                                                              merge_duplicate_headers=True,
                                                              style_as_list_view=True,
                                                              css=[
                                                                  {
                                                                      'selector': '.previous-page, .next-page, '
                                                                                  '.first-page, .last-page',
                                                                      'rule': 'color: #2c8cff'}
                                                              ])
                            ),
                        ]
                    ),
                    html.Br(id='after_loading', hidden=False),
                ],
            ),
            html.Div(
                id="submit-btn-outer",
                hidden=True,
                children=[
                    html.Br(),
                    html.Button(id="submit-btn", children="Submit & Next", n_clicks=0,
                                style={'width': '100%', 'color': 'white'},
                                disabled=False),
                ],
            ),
            html.Br(),
            html.Div(
                id="download-outer",
                hidden=initialize_download_button(annotated_list),
                children=[
                    html.Button(id="download-btn", children="Download current annotations.zip", n_clicks=0,
                                style={'width': '100%', 'color': 'white'},
                                disabled=False),
                    dcc.Download(id="download-annotations")
                ]
            ),
        ],
        style={'width': '100%', 'color': 'black'}
    )


def generate_all_patients_graph(labitem, **kwargs):
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
                family=kwargs['config']['title-font'],
                size=kwargs['config']['title-size'],
                color=kwargs['config']['title-color']
            )},
        xaxis_title=f"{labitemsid_dict[labitem]} ({units})",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(
            family=kwargs['config']['text-font'],
            size=kwargs['config']['text-size'],
            color=kwargs['config']['text-color']
        ),
        height=kwargs['config']['height'],
        margin=dict(l=50, r=50, t=90, b=20),
    )
    return fig


def generate_tab_graph(labitem, patient, template_labitems, **kwargs):
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

    if kwargs['config']['spikes']:
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
                family=kwargs['config']['title-font'],
                size=kwargs['config']['title-size'],
                color=kwargs['config']['title-color']
            )},
        xaxis_title="Time (Hours)",
        font=dict(
            family=kwargs['config']['text-font'],
            size=kwargs['config']['text-size'],
            color=kwargs['config']['text-color']
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95),
        margin=dict(l=50, r=0, t=90, b=20),
        height=kwargs['config']['height']
    )
    return fig


def query_patients(labitem):
    list_of_labitems = [labitem, bg_pair[0], bg_pair[1], chem_pair[0], chem_pair[1], cbc_pair[0], cbc_pair[1]]
    list_of_patient_sets = []
    for each_labitem in list_of_labitems:
        table = df_labevents.query(f'itemid == {each_labitem}')
        list_of_patient_sets.append(set(table['subject_id'].unique()))

    map_reduce_list = []
    for i in range(0, len(list_of_patient_sets)):
        for each_item in list_of_patient_sets[i]:
            map_reduce_list.append([each_item, i])

    dict_grouped = {}
    for each_pair in map_reduce_list:
        if each_pair[0] not in dict_grouped:
            dict_grouped[each_pair[0]] = each_pair[1:]
        else:
            dict_grouped[each_pair[0]].append(each_pair[1])

    second_map_reduce_list = []
    for each_key in dict_grouped:
        if len(dict_grouped[each_key]) > 1:
            second_map_reduce_list.append([tuple(dict_grouped[each_key]), each_key])

    ranked_patients = sorted(second_map_reduce_list, key=lambda x: (len(x[0]), x[1]))[::-1]
    ranked_patients = [patient for patient in ranked_patients if 0 in patient[0]]
    list_of_patients_ranked = [subject_id[1] for subject_id in ranked_patients]

    patients = [{"label": each_patient, "value": each_patient} for each_patient in
                list_of_patients_ranked]
    return patients


def initialize_download_button(annotated):
    if annotated:
        return False
    return True


def initialize_upload_field(config_exists):
    if config_exists:
        return True
    return False


def initialize_all_patients_graph():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    fig = generate_all_patients_graph(labitems[0]["value"], config=config['graphs']['kwargs'])
    return fig


def initialize_tab_graph(pair):
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(labitems[0]["value"])
    if not patients[0]:
        return {}
    first_patient = patients[0]["value"]
    fig = generate_tab_graph(labitems[0]["value"], first_patient, template_labitems=pair,
                             config=config['graphs']['kwargs'])
    return fig


def initialize_tab():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(labitems[0]["value"])
    if not patients[0]:
        return True
    first_patient = patients[0]["value"]
    disabled = update_graph(labitems[0]["value"], first_patient, submit=0)[2]
    return disabled


def initialize_labitem_select():
    options = [
        {"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} if each_id in unannotated_list else
        {"label": f'{each_id}: {labitemsid_dict[each_id]} ✔', "value": each_id} for each_id in labitemsid_dict
    ]
    first_labitem = unannotated_list[0]
    return options, first_labitem


def initialize_patient_select():
    labitems = [{"label": f'{each_id}: {labitemsid_dict[each_id]}', "value": each_id} for each_id in labitemsid_dict]
    options = query_patients(labitems[0]["value"])
    if not options:
        return [], None
    first_labitem = options[0]["value"]
    return options, first_labitem


def initialize_annotate_select():
    loinc_codes = [{"label": f'{each_code}: {loinc_dict[each_code]}', "value": each_code} for each_code in loinc_dict]
    if config['temp']['test']:
        return loinc_codes[2000:4000]
    return loinc_codes


def annotate(labitem, annotation):
    # labitem_row = df_labitems.query(f'itemid == {labitem}')
    labitem_dict = {'itemid': labitem,
                    'label': labitemsid_dict[labitem],
                    # 'mimic_loinc': labitem_row['loinc_code'].item(),      # not available in mimic-iv-v2.0, removed in d-labitems
                    'loincid': annotation,
                    'loinclabel': loinc_dict[annotation]
                    }

    filename = os.path.join(PATH_results, f"{labitem}.json")
    with open(filename, "w") as outfile:
        json.dump(labitem_dict, outfile, indent=4)

    if labitem in unannotated_list:
        unannotated_list.remove(labitem)
    return


def update_labitem_options(options, labitem):
    if unannotated_list:
        next_value = unannotated_list[0]
    else:
        next_value = options[0]['value']
    for index in range(len(options)):
        if options[index]['value'] == labitem:
            # del options[index]
            options[index]['label'] = f'{labitem}: {labitemsid_dict[labitem]} ✔'
            if unannotated_list:
                for next_index in range(index, len(options)):
                    if options[next_index]['value'] in unannotated_list:
                        next_value = options[next_index]['value']
                        break
                    if next_index == len(options) - 1:
                        next_value = unannotated_list[0]
            break
    return options, next_value


class ScorerNotAvailble(Exception):
    pass


######################################################################################################
@app.callback(
    Output('submit-btn-outer', 'hidden'),
    [
        Input('annotate-select', 'value'),
    ]
)
def enable_submit_button(annotation):
    if annotation:
        return False
    return True


@app.callback(
    Output('download-outer', 'hidden'),
    [
        Input('submit-btn', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def enable_download_button(n_clicks):
    return False


@app.callback(
    Output('download-annotations', 'data'),
    [
        Input('download-btn', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def download_annotations(n_clicks):
    def write_archive(bytes_io):
        with ZipFile(bytes_io, mode="w") as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(config['directories']['results']):
                for filename in filenames:
                    # Add file to zip
                    zipObj.writestr(filename, os.path.basename(filename))

    return dcc.send_bytes(write_archive, "annotations.zip")


@app.callback(
    Output('annotate-select', 'value'),
    # Output('annotate-text', 'value'),
    Output('confirm-replace', 'displayed'),
    Output("related-datatable", "active_cell"),
    Output("related-datatable", "selected_cells"),
    Output("loinc-datatable", "active_cell"),
    Output("loinc-datatable", "selected_cells"),
    [
        Input("submit-btn", "n_clicks"),
        Input('loinc-datatable', 'active_cell'),
        Input('annotate-select', 'value'),
    ],
    [
        State('labitem-select', 'value'),
        # State('annotate-text', 'value')
        State("loinc-datatable", "data"),
    ]
)
def reset_annotation(n_clicks, replace_annotation, annotation, labitem, curr_data):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        annotate(labitem, annotation)
        return '', False, None, [], None, []
    elif triggered_id == 'loinc-datatable.active_cell':
        if replace_annotation['row'] == 1:
            new_annotation = curr_data[replace_annotation['row']]['LOINC_NUM']
            return new_annotation, False, None, [], None, []
        else:
            raise PreventUpdate
    elif triggered_id == 'annotate-select.value':
        if not annotation:
            return '', False, None, [], None, []
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


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
    if triggered_id == '.':
        raise PreventUpdate
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

    triggered_id = dash.callback_context.triggered[0]['prop_id']
    # if triggered_id == 'next-btn.n_clicks':
    #     return

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
        list_of_labitems = [bg_pair[0], bg_pair[1], chem_pair[0], chem_pair[1], cbc_pair[0], cbc_pair[1]]
        labitem_exists = []
        tabs = []
        for each_labitem in list_of_labitems:
            table = df_labevents.query(f'itemid == {each_labitem}')
            patients_with_labitem = set(table['subject_id'].unique())
            if patient in patients_with_labitem:
                labitem_exists.append(True)
            else:
                labitem_exists.append(False)
        for i in range(0, len(labitem_exists), 2):
            if labitem_exists[i] and labitem_exists[i + 1]:
                tabs.append(True)
            else:
                tabs.append(False)
        disabled = False
        if tabs[0]:
            tab_bg = (generate_tab_graph(labitem, patient, bg_pair, config=config['graphs']['kwargs']), disabled)
        else:
            tab_bg = ({}, True)
        if tabs[1]:
            tab_chem = (generate_tab_graph(labitem, patient, chem_pair, config=config['graphs']['kwargs']), disabled)
        else:
            tab_chem = ({}, True)
        if tabs[2]:
            tab_cbc = (generate_tab_graph(labitem, patient, cbc_pair, config=config['graphs']['kwargs']), disabled)
        else:
            tab_cbc = ({}, True)
        tab_labitem = generate_all_patients_graph(labitem, config=config['graphs']['kwargs'])

        return tab_labitem, tab_bg[0], tab_bg[1], tab_chem[0], tab_chem[1], tab_cbc[0], tab_cbc[1]
    return generate_all_patients_graph(labitem,
                                       config=config['graphs']['kwargs']), {}, disabled, {}, disabled, {}, disabled


@app.callback(
    Output("annotation-outer", "hidden"),
    [
        Input("annotate-select", "value"),
    ],
)
def show_related_outer(annotation):
    if annotation:
        return False
    else:
        return True


@app.callback(
    Output("loinc-results-outer", "hidden"),
    Output("loinc-datatable", "data"),
    Output("loinc-datatable", "columns"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input('related-datatable', 'active_cell'),
    ],
    [
        State("related-datatable", "data"),
    ]
)
def update_loinc_datatable(annotation, submit, related, curr_data):
    triggered_ids = dash.callback_context.triggered
    if triggered_ids[0]['prop_id'] == 'submit-btn.n_clicks' or not annotation:
        return True, None, []
    df_data = df_loinc.loc[df_loinc['LOINC_NUM'] == annotation]
    data = df_data.to_dict('records')
    columns = [{"name": i, "id": i} for i in df_data.columns]
    if related:
        df_data = pd.concat([df_data, df_loinc.loc[
            df_loinc['LOINC_NUM'] == [each_key for each_key in curr_data if each_key['id'] == related['row_id']][0][
                'LOINC_NUM']]])
        data = df_data.to_dict('records')
    return False, data, columns


@app.callback(
    Output("related-datatable-outer", "hidden"),
    Output("related-datatable", "data"),
    Output("before_loading", "hidden"),
    Output("after_loading", "hidden"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_related_datatable(annotation, submit):
    if not annotation:
        return True, None, False, False

    triggered_ids = dash.callback_context.triggered
    if triggered_ids[0]['prop_id'] == 'submit-btn.n_clicks':
        return True, None, False, False

    if config['loinc']['related']['scorer'] == 'partial_ratio':
        query = list(df_loinc_new.loc[df_loinc_new['LOINC_NUM'] == annotation]['LONG_COMMON_NAME'])[0]
        choices = list(df_loinc_new['LONG_COMMON_NAME'])
        related = process.extractBests(query, choices, scorer=fuzz.partial_ratio, limit=100, score_cutoff=85)
        data = df_loinc_new[df_loinc_new['LONG_COMMON_NAME'].isin([i[0] for i in related[1:]])].to_dict('records')
    elif config['loinc']['related']['scorer'] == 'jaro_winkler':
        query = list(df_loinc_new.loc[df_loinc_new['LOINC_NUM'] == annotation]['LONG_COMMON_NAME'])[0]
        choices = list(df_loinc_new['LONG_COMMON_NAME'])
        related = process.extractBests(query, choices, scorer=jaro.jaro_winkler_metric, limit=100, score_cutoff=85)
        data = df_loinc_new[df_loinc_new['LONG_COMMON_NAME'].isin([i[0] for i in related[1:]])].to_dict('records')
    elif config['loinc']['related']['scorer'] == 'tf-idf':
        # NLP: tf-idf
        query = vectorizer.transform([df_loinc_new.loc[df_loinc_new['LOINC_NUM'] == annotation]['LONG_COMMON_NAME']])
        scores = cosine_similarity(tf_idf_matrix, query)
        df_loinc_new_temp = df_loinc_new
        df_loinc_new_temp['cosine_score'] = scores
        df_loinc_new_temp = df_loinc_new_temp.sort_values(by=['cosine_score'], ascending=False)
        data = df_loinc_new_temp[1:101].to_dict('records')
    else:
        raise ScorerNotAvailble("Please define scorer from available options in the configuration file.")
    return False, data, True, True


# @app.callback(
#     Output("annotate-select", "options"),
#     [
#         Input("annotate-select", "search_value"),
#         Input("submit-btn", "n_clicks"),
#     ],
# )
# def update_options(search_value, submit):
#     if not search_value:
#         raise PreventUpdate
#     return [o for o in annotation_options if search_value in o["label"].lower()]
#
#
# # For dynamic dropdown, also make sure to remove 'options' tag in annotate-select declaration
# annotation_options = initialize_annotate_select()

######################################################################################################
app.layout = html.Div(
    id="app-container",
    children=[
        # Replace confirmation
        html.Div(
            id="replace-confirmation",
            children=[
                dcc.ConfirmDialog(
                    id='confirm-replace',
                    message='This measurement has already been annotated. Are you sure you want to replace the '
                            'existing annotation?',
                )]
        ),
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
                                            style={'height': '375px'},
                                            id="all_patients_graph",
                                            figure=initialize_all_patients_graph()
                                        )
                                    ]),
                            dcc.Tab(label=config['graphs']['pairs']['pair_one']['label'], id="blood_gas_tab",
                                    disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '375px'},
                                            id="blood_gas_graph",
                                            figure=initialize_tab_graph(bg_pair)
                                        )
                                    ]),
                            dcc.Tab(label=config['graphs']['pairs']['pair_two']['label'], id="chemistry_tab",
                                    disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '375px'},
                                            id="chemistry_graph",
                                            figure=initialize_tab_graph(chem_pair)
                                        )
                                    ]),
                            dcc.Tab(label=config['graphs']['pairs']['pair_three']['label'], id="cbc_tab",
                                    disabled=initialize_tab(),
                                    style={'color': '#1a75f9'},
                                    selected_style={
                                        'color': '#1a75f9',
                                        'border-width': '3px'
                                    },
                                    children=[
                                        dcc.Graph(
                                            style={'height': '375px'},
                                            id="cbc_graph",
                                            figure=initialize_tab_graph(cbc_pair)
                                        )
                                    ]),
                        ], id='tabs', value='home-tab'),
                        html.Br(),
                        html.Div(id='loinc-results-outer',
                                 hidden=True,
                                 children=[
                                     html.P(children=[
                                         html.B('Compare Results (click to swap search terms):'),
                                     ]),
                                     html.Div(
                                         id="loinc-datatable-outer",
                                         className='loinc-datatable',
                                         hidden=False,
                                         children=[
                                             dash_table.DataTable(id='loinc-datatable',
                                                                  data=None,
                                                                  columns=[],
                                                                  style_data={
                                                                      'whiteSpace': 'normal',
                                                                      'height': 'auto',
                                                                      'lineHeight': '15px',
                                                                  },
                                                                  style_table={
                                                                      'height': 'auto',
                                                                      'overflowY': 'auto'
                                                                  },
                                                                  style_cell={
                                                                      'textAlign': 'left',
                                                                      'backgroundColor': 'transparent',
                                                                      'color': 'black'
                                                                  },
                                                                  style_header={
                                                                      'fontWeight': 'bold',
                                                                      'color': '#2c8cff'
                                                                  },
                                                                  style_data_conditional=[
                                                                      {
                                                                          'if': {
                                                                              'state': 'active'  # 'active' | 'selected'
                                                                          },
                                                                          'backgroundColor': 'transparent',
                                                                          'border': '1px solid lightgray'
                                                                      }],
                                                                  page_size=10,
                                                                  )
                                         ]
                                     )]
                                 ),
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
