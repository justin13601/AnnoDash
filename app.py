#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""

import os
import io
import re
import csv
import time
import json
import yaml
import errno
import base64
from datetime import timedelta, datetime as dt
from collections import defaultdict
from ml_collections import config_dict
from zipfile import ZipFile

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate
from flask import Flask, send_file

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import jaro
import pickle
import shelve
from fuzzywuzzy import fuzz, process
from ftfy import fix_text
from google.cloud import bigquery

from related_ontologies.related import ngrams, generateRelatedOntologies, TfidfVectorizer, cosine_similarity


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
# APP #
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

app.title = "MIMIC-Dash: A Clinical Terminology Annotation Dashboard"
server = app.server
app.config.suppress_callback_exceptions = True


######################################################################################################
# DATA #
######################################################################################################
# callback_manager.attach_to_app(app)

class ScorerNotAvailable(Exception):
    pass


class ConfigurationFileError(Exception):
    pass


class OntologyNotSupported(Exception):
    pass


def load_items(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    items = pd.read_csv(path)
    dictionary = pd.Series(items[items.columns[1]].values, index=items[items.columns[0]].values).to_dict()
    print('Done.\n')
    return items, dictionary


def load_data(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    data = pd.read_csv(path)
    # Date, format charttime
    data["charttime"] = data["charttime"].apply(
        lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
    )  # String -> Datetime
    print('Done.\n')
    return data


# load data
def load_ontology(ontology):
    if ontology == 'loinc':
        path = os.path.join(PATH_ontology, config.ontology.filename)
        filename = os.path.basename(path).strip()
        print(f'Loading {filename}...')
        data = pd.read_csv(path, dtype=object)
        dictionary = pd.Series(data.LONG_COMMON_NAME.values, index=data.LOINC_NUM.values).to_dict()
        print(f"LOINC codes (CLASSTYPE={config.ontology.class_value}, "
              f"{config.ontology.class_label}) loaded and processed.\n")
    elif ontology == 'snomed':
        path = os.path.join(PATH_ontology, config.ontology.filename)
        filename = os.path.basename(path).strip()
        print(f'Loading {filename}...')
        data = pd.read_csv(path, sep='\t')
        dictionary = pd.Series(data.term.values, index=data.conceptId.values).to_dict()
        print(f"SNOMED-CT codes (Hierarchy={config.ontology.class_value}, "
              f"{config.ontology.class_label}) loaded and processed.\n")
    else:
        raise OntologyNotSupported
    print('Done.\n')
    return data, dictionary


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.unsafe_load(f)
        print('Done.\n')
        return configurations


def load_annotations(path):
    annotation_files = [each_json for each_json in os.listdir(path) if each_json.endswith('.json')]
    annotated = [int(each_item.strip('.json')) for each_item in annotation_files]
    skipped = []
    for each_file in annotation_files:
        with open(os.path.join(path, each_file)) as jsonFile:
            data = json.load(jsonFile)
            if 'skipped' in data['annotatedid'].lower().strip():
                skipped.append(int(each_file.strip('.json')))
    return annotated, skipped


def download_annotation(annotation):
    results_folder = 'results-json'
    path = os.path.join(results_folder, f"{annotation}.json")
    return send_file(path, as_attachment=True)


config_file = 'config.yaml'
if os.path.exists(config_file):
    print('Configuration file found.')
    config = load_config(config_file)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

# paths
PATH_data = config.directories.data.location
PATH_items = config.directories.concepts.location
PATH_results = config.directories.results
PATH_ontology = config.ontology.location
PATH_related = config.ontology.related.location

if 'demo-data' in PATH_data:
    print("Demo data selected.")

df_events = load_data(os.path.join(PATH_data, config.directories.data.filename))

df_items, itemsid_dict = load_items(os.path.join(PATH_items, config.directories.concepts.filename))
print("Data loaded.\n")

df_ontology, ontology_dict = load_ontology(ontology=config.ontology.name)
df_ontology_new = pd.DataFrame(
    {'CODE': list(ontology_dict.keys()), 'LABEL': list(ontology_dict.values())})
df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})
print('Ontology loaded.\n')

# load tf_idf matrix if chosen as scorer:
if config.ontology.related.scorer == 'tf_idf':
    with shelve.open(os.path.join(PATH_related, 'tf_idf.shlv'), protocol=5) as shlv:
        ngrams = shlv['ngrams']
        vectorizer = shlv['model']
        tf_idf_matrix = shlv['tf_idf_matrix']

annotated_list, skipped_list = load_annotations(PATH_results)
unannotated_list = list(set(itemsid_dict.keys()) - set(annotated_list))
unannotated_list.sort()

# define item pairs for patient specific tabs
pairs = []
for each in config.graphs.pairs.values():
    pairs.append((each['label'], each['item_1'], each['item_2']))
bg_pair = pairs[0][1:]  # Default PO2 & PCO2, Blood; Could add FiO2
chem_pair = pairs[1][1:]  # Default Creatinine & Potassium, Blood; Could also use Sodium & Glucose (overlay 4?)
cbc_pair = pairs[2][1:]  # Default Hemoglobin & WBC, Blood; Could add RBC (the one with space) -> no observations :(


######################################################################################################
# FUNCTIONS #
######################################################################################################
def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5(""),
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
            html.Div(
                id='item-copy-outer',
                hidden=False,
                children=[
                    dcc.Clipboard(
                        id='item-copy',
                        title="Copy Concept/Item",
                        style={
                            "color": "#c9ddee",
                            "fontSize": 15,
                            "verticalAlign": "center",
                            'float': 'right',
                            'margin': 'auto'
                        },
                    )
                ]
            ),
            html.P("Select Concept:"),
            dcc.Dropdown(
                id="item-select",
                clearable=False,
                value=initialize_item_select()[1],
                style={"border-radius": 0},
                options=initialize_item_select()[0],
            ),
            html.Br(),
            html.Div(
                id='patient-copy-outer',
                hidden=False,
                children=[
                    dcc.Clipboard(
                        id='patient-copy',
                        title="Copy Patient ID",
                        style={
                            "color": "#c9ddee",
                            "fontSize": 15,
                            "verticalAlign": "center",
                            'float': 'right',
                            'margin': 'auto'
                        },
                    )
                ]
            ),
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
            html.Div(
                id='annotate-copy-outer',
                hidden=True,
                children=[
                    dcc.Clipboard(
                        id='annotate-copy',
                        title="Copy Annotation",
                        style={
                            "color": "#c9ddee",
                            "fontSize": 15,
                            "verticalAlign": "center",
                            'float': 'right',
                            'margin': 'auto'
                        },
                    )
                ]
            ),
            html.P("Select Annotation:"),
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
                            dcc.Clipboard(
                                id='related-copy',
                                title="Copy Related Results",
                                style={
                                    "color": "#c9ddee",
                                    "fontSize": 15,
                                    "verticalAlign": "center",
                                    'float': 'right',
                                    'margin': 'auto'
                                },
                            ),
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
                                                                  {'name': "",
                                                                   'id': 'RELEVANCE'},
                                                                  {'name': "CODE",
                                                                   'id': 'CODE'},
                                                                  {'name': "LABEL",
                                                                   'id': 'LABEL'}
                                                              ],
                                                              sort_action='native',
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
                                                                      'if': {'column_id': 'CODE'},
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
            html.Div(
                id="skip-outer",
                hidden=False,
                children=[
                    html.Br(),
                    dcc.RadioItems(
                        id='skip-radio-items',
                        options=['Unsure', 'Invalid Source Data', 'Other'],
                        value='Unsure',
                        style={'width': '100%', 'color': 'white', 'textAlign': 'center', 'verticalAlign': 'center'},
                        labelStyle={'margin-right': '30px'},
                        inline=True,
                    ),
                    html.Div(id='skip-other-outer',
                             hidden=True,
                             children=[
                                 dcc.Input(
                                     id="skip-other-text",
                                     placeholder="Reason...",
                                     debounce=True,
                                     style={"width": '100%'},
                                     autoFocus=True,
                                 ),
                             ]),
                    html.Button(id="skip-btn", children="Skip", n_clicks=0,
                                style={'width': '100%', 'color': 'white', 'display': 'inline-block'},
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
            html.Br(),
        ],
        style={'width': '100%', 'color': 'black'}
    )


def generate_all_patients_graph(item, **kwargs):
    table = df_events.query(f'itemid == {item}')
    if table.empty:
        return {}
    df_temp = pd.to_numeric(table['value'], errors='coerce')
    if df_temp.isna().sum().sum() / df_temp.shape[0] > 0.5:
        table['value'] = table['value'].str.upper()
        table = table.groupby(['value'])['value'].count()
        df_data = pd.DataFrame(table)
        units = ''
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Count",
            x=df_data.index,
            y=df_data["value"],
            hovertemplate=
            '%{y}' +
            '<br>Percent: %{text}',
            text=round((df_data["value"] / sum(df_data["value"]) * 100), 2).astype(str) + '%',
        ))
        fig.update_xaxes(showspikes=True, spikecolor="black", spikethickness=1, spikedash='dot',
                         spikemode='across+marker')
        fig.update_layout(hovermode="x unified")
        fig.update_traces(marker_color='rgb(100,169,252)', hovertemplate=None)
        ylabel = 'Count'
    else:  # numerical data
        table.replace(np.inf, np.nan)
        table['value'] = pd.to_numeric(table['value'], errors='coerce')
        table.dropna(subset=['value'], inplace=True)
        hist_data = [list(table['value'])]
        if hist_data == [[]]:
            return {}
        units = f"({list(table['valueuom'])[0]})"
        group_labels = [f"{itemsid_dict[item]} (%)"]
        fig = ff.create_distplot(hist_data, group_labels, colors=['rgb(44,140,255)'])
        ylabel = ''

    df_temp = df_items.query(f'itemid == {item}').dropna(axis=1, how='all')
    if len(df_temp.columns) > 2:
        row_dict = df_items.query(f'itemid == {item}').iloc[:, 2:].to_dict(orient='records')[0]
        metadata = ', '.join([f'{key.capitalize()}: {value}' for key, value in row_dict.items()])
        title_template = {
            'text': f"{itemsid_dict[item]}<br><sup>"
                    f"{metadata}</sup>",
            'y': 0.905,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family=kwargs['config'].title_font,
                size=kwargs['config'].title_size,
                color=kwargs['config'].title_color
            )}
    else:
        title_template = {
            'text': f"{itemsid_dict[item]}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family=kwargs['config'].title_font,
                size=kwargs['config'].title_size,
                color=kwargs['config'].title_color
            )}
    fig.update_layout(
        title=title_template,
        xaxis_title=f"{itemsid_dict[item]} {units}",
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
        font=dict(
            family=kwargs['config'].text_font,
            size=kwargs['config'].text_size,
            color=kwargs['config'].text_color
        ),
        height=kwargs['config'].height,
        margin=dict(l=50, r=50, t=90, b=20),
    )
    return fig


def generate_tab_graph(item, patient, template_items, **kwargs):
    table_item_1 = df_events.query(f'itemid == {template_items[0]}')
    table_item_1['value'] = pd.to_numeric(table_item_1['value'], errors='coerce')
    table_item_2 = df_events.query(f'itemid == {template_items[1]}')
    table_item_2['value'] = pd.to_numeric(table_item_2['value'], errors='coerce')
    table_item_target = df_events.query(f'itemid == {item}')
    table_item_target['value'] = pd.to_numeric(table_item_target['value'], errors='coerce')
    table_item_patient_1 = table_item_1.query(f'subject_id == {patient}')
    table_item_patient_2 = table_item_2.query(f'subject_id == {patient}')
    table_item_patient_target = table_item_target.query(f'subject_id == {patient}')
    units_1 = list(table_item_1['valueuom'])[0]
    units_2 = list(table_item_2['valueuom'])[0]
    units_target = list(table_item_target['valueuom'])[0]

    if table_item_patient_1.empty or table_item_patient_2.empty or table_item_patient_target.empty:
        return {}

    start_date = min(min(table_item_patient_target['charttime']),
                     min(table_item_patient_target['charttime'])) - timedelta(hours=12)
    end_date = start_date + timedelta(hours=96) + timedelta(hours=12)

    mask_1 = (table_item_patient_1['charttime'] > start_date) & (table_item_patient_1['charttime'] <= end_date)
    table_item_patient_1 = table_item_patient_1.loc[mask_1]

    mask_2 = (table_item_patient_2['charttime'] > start_date) & (table_item_patient_2['charttime'] <= end_date)
    table_item_patient_2 = table_item_patient_2.loc[mask_2]

    mask_plot = (table_item_patient_target['charttime'] > start_date) & (
            table_item_patient_target['charttime'] <= end_date)
    table_item_patient_target = table_item_patient_target.loc[mask_plot]

    series_names = [itemsid_dict[template_items[0]], itemsid_dict[template_items[1]],
                    itemsid_dict[item]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=table_item_patient_1["charttime"], y=table_item_patient_1["value"],
                   name=f"{series_names[0]} ({units_1})"), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_item_patient_2["charttime"], y=table_item_patient_2["value"],
                   name=f"{series_names[1]} ({units_2})"), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=table_item_patient_target["charttime"], y=table_item_patient_target["value"],
                   name=f"{series_names[2]} ({units_target})"),
        secondary_y=True,
    )

    fig.update_traces(mode="markers+lines", hovertemplate='Value: %{y:.1f}<extra></extra>')

    if kwargs['config'].spikes:
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
            'text': f"{itemsid_dict[item]}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family=kwargs['config'].title_font,
                size=kwargs['config'].title_size,
                color=kwargs['config'].title_color
            )},
        xaxis_title="Time (Hours)",
        font=dict(
            family=kwargs['config'].text_font,
            size=kwargs['config'].text_size,
            color=kwargs['config'].text_color
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.95, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=50, r=0, t=90, b=20),
        height=kwargs['config'].height
    )
    return fig


def query_patients(item):
    list_of_items = [item, bg_pair[0], bg_pair[1], chem_pair[0], chem_pair[1], cbc_pair[0], cbc_pair[1]]
    list_of_patient_sets = []
    for each_item in list_of_items:
        table = df_events.query(f'itemid == {each_item}')
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
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    fig = generate_all_patients_graph(items[0]["value"], config=config.graphs.kwargs)
    return fig


def initialize_tab_graph(pair):
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(items[0]["value"])
    if not patients[0]:
        return {}
    first_patient = patients[0]["value"]
    fig = generate_tab_graph(items[0]["value"], first_patient, template_items=pair,
                             config=config.graphs.kwargs)
    return fig


def initialize_tab():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(items[0]["value"])
    if not patients[0]:
        return True
    first_patient = patients[0]["value"]
    disabled = update_graph(items[0]["value"], first_patient, submit=0)[2]
    return disabled


def initialize_item_select():
    options = [
        {"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} if each_id in unannotated_list else
        {"label": f'{each_id}: {itemsid_dict[each_id]} ✔', "value": each_id} if each_id in list(
            set(annotated_list) - set(skipped_list)) else
        {"label": f'{each_id}: {itemsid_dict[each_id]} ⚠', "value": each_id} for each_id in itemsid_dict
    ]
    first_item = unannotated_list[0]
    return options, first_item


def initialize_patient_select():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in itemsid_dict]
    options = query_patients(items[0]["value"])
    if not options:
        return [], None
    first_item = options[0]["value"]
    return options, first_item


def initialize_annotate_select():
    ontology_codes = [{"label": f'{each_code}: {ontology_dict[each_code]}', "value": each_code} for each_code in
                      ontology_dict]
    if config.temp.five_percent_dataset:  # for testing
        target_codes_for_demo = ['2069-3', '1959-6', '5767-9', '5778-6']  # bicarbonate, chloride, urine color and app.
        ontology_codes_temp_for_demo = [{"label": f'{each_code}: {ontology_dict[each_code]}', "value": each_code} for
                                        each_code in ontology_dict if each_code in target_codes_for_demo]
        ontology_codes_temp_for_demo_2 = ontology_codes[2000:4000]
        for aLis1 in ontology_codes_temp_for_demo:
            if aLis1 not in ontology_codes_temp_for_demo_2:
                ontology_codes_temp_for_demo_2.append(aLis1)
        return ontology_codes_temp_for_demo_2
        # return ontology_codes[2000:4000] 
    return ontology_codes


def annotate(item, annotation, skipped=False):
    if skipped:
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     'ontology': config.ontology.name,
                     'annotatedid': f'Skipped: {annotation}',
                     'annotatedlabel': 'n/a'
                     }
    else:
        # item_row = df_items.query(f'itemid == {item}')
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     # 'mimic_loinc': item_row['loinc_code'].item(),      # not available in mimic-iv-v2.0, removed in d-items
                     'annotatedid': annotation,
                     'annotatedlabel': ontology_dict[annotation]
                     }
    filename = os.path.join(PATH_results, f"{item}.json")
    with open(filename, "w") as outfile:
        json.dump(item_dict, outfile, indent=4)
    return


def update_item_options(options, item, skipped=False):
    new_value = item
    value_location = list(itemsid_dict.keys()).index(item)
    if skipped:
        options[value_location]['label'] = f'{item}: {itemsid_dict[item]} ⚠'
    else:
        options[value_location]['label'] = f'{item}: {itemsid_dict[item]} ✔'
    if unannotated_list:
        for i in range(value_location, len(options)):
            if i == len(options) - 1:
                new_value = unannotated_list[0]
            elif options[i]['value'] in unannotated_list:
                new_value = options[i]['value']
                break
    elif skipped_list:
        for i in range(value_location, len(options)):
            if i == len(options) - 1:
                new_value = skipped_list[0]
            elif options[i]['value'] in skipped_list:
                new_value = options[i]['value']
                break
    else:
        new_value = options[0]['value']
    return options, new_value


######################################################################################################
# CALLBACKS #
######################################################################################################
@app.callback(
    Output("item-copy", "content"),
    [
        Input("item-copy", "n_clicks"),
    ],
    [
        State("item-select", "value"),
    ]
)
def copy_item(_, item):
    if item is None:
        raise PreventUpdate
    clipboard = f'{item}: {itemsid_dict[item]}'
    return clipboard


@app.callback(
    Output("patient-copy", "content"),
    [
        Input("patient-copy", "n_clicks"),
    ],
    [
        State("patient-select", "value"),
    ]
)
def copy_patient(_, patient):
    if patient is None:
        raise PreventUpdate
    return str(patient)


@app.callback(
    Output("annotate-copy", "content"),
    [
        Input("annotate-copy", "n_clicks"),
    ],
    [
        State("annotate-select", "value"),
    ]
)
def copy_annotation(_, annotation):
    if annotation is None:
        raise PreventUpdate
    clipboard = ontology_dict[annotation]
    return clipboard


@app.callback(
    Output("related-copy", "content"),
    [
        Input("related-copy", "n_clicks"),
    ],
    [
        State("related-datatable", "data"),
    ]
)
def copy_related_datatable(_, data):
    dff = pd.DataFrame(data)
    return dff.to_csv(index=False)


@app.callback(
    Output("ontology-copy", "content"),
    [
        Input("ontology-copy", "n_clicks"),
    ],
    [
        State("ontology-datatable", "data"),
    ]
)
def copy_ontology_datatable(_, data):
    dff = pd.DataFrame(data)
    return dff.to_csv(index=False)


@app.callback(
    Output("item-copy-outer", "hidden"),
    Output("patient-copy-outer", "hidden"),
    Output("annotate-copy-outer", "hidden"),
    [
        Input("item-select", "value"),
        Input("patient-select", "value"),
        Input("annotate-select", "value"),
    ]
)
def show_hide_clipboard(item, patient, annotation):
    show_item = True
    show_patient = True
    show_annotate = True
    if item:
        show_item = False
    if patient:
        show_patient = False
    if annotation:
        show_annotate = False

    return show_item, show_patient, show_annotate


@app.callback(
    Output("submit-btn-outer", "hidden"),
    [
        Input("annotate-select", "value"),
    ]
)
def enable_submit_button(annotation):
    if annotation:
        return False
    return True


@app.callback(
    Output("download-outer", "hidden"),
    [
        Input("submit-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def enable_download_button(n_clicks):
    return False


@app.callback(
    Output("skip-other-outer", "hidden"),
    [
        Input("skip-radio-items", "value"),
    ],
    prevent_initial_call=True,
)
def enable_download_button(value):
    if value == 'Other':
        return False
    return True


@app.callback(
    Output("skip-other-text", "value"),
    Output("skip-radio-items", "value"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def clear_reset_skip(submit_n_clicks, skip_n_clicks):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return '', "Unsure"
    elif triggered_id == 'skip-btn.n_clicks':
        return '', "Unsure"
    else:
        raise PreventUpdate


@app.callback(
    Output("download-annotations", "data"),
    [
        Input("download-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def download_annotations(n_clicks):
    def write_archive(bytes_io):
        with ZipFile(bytes_io, mode="w") as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(config.directories.results):
                for filename in filenames:
                    # Add file to zip
                    zipObj.writestr(filename, os.path.basename(filename))

    return dcc.send_bytes(write_archive, "annotations.zip")


@app.callback(
    Output("annotate-select", "value"),
    # Output('annotate-text', 'value'),
    Output("confirm-replace", "displayed"),
    Output("related-datatable", "active_cell"),
    Output("related-datatable", "selected_cells"),
    Output("ontology-datatable", "active_cell"),
    Output("ontology-datatable", "selected_cells"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("ontology-datatable", "active_cell"),
        Input("annotate-select", "value"),
    ],
    [
        State("item-select", "value"),
        # State('annotate-text', 'value')
        State("ontology-datatable", "data"),
        State("skip-radio-items", "value"),
        State("skip-other-text", "value"),
    ]
)
def reset_annotation(submit_n_clicks, skip_n_clicks, replace_annotation, annotation, item, curr_data, skip_reason,
                     other_reason):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        annotate(item, annotation)
        return '', False, None, [], None, []
    elif triggered_id == 'skip-btn.n_clicks':
        if skip_reason == 'Other':
            annotate(item, other_reason, skipped=True)
        else:
            annotate(item, skip_reason, skipped=True)
        return '', False, None, [], None, []
    elif triggered_id == 'ontology-datatable.active_cell':
        if replace_annotation['row'] == 1:
            new_annotation = list(curr_data[replace_annotation['row']].values())[0]
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
    Output("item-select", "options"),
    Output("item-select", "value"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
    ],
    [
        State("item-select", "options"),
        State("item-select", "value"),
    ]
)
def update_item_dropdown(submit_n_clicks, skip_n_clicks, options, value):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == '.':
        raise PreventUpdate
    new_value = value
    if triggered_id == 'submit-btn.n_clicks':
        if value in unannotated_list:
            unannotated_list.remove(value)
        options, new_value = update_item_options(options, value)
    elif triggered_id == 'skip-btn.n_clicks':
        if value in unannotated_list:
            unannotated_list.remove(value)
        skipped_list.append(value)
        options, new_value = update_item_options(options, value, skipped=True)
    return options, new_value


@app.callback(
    Output("patient-select", "options"),
    Output("patient-select", "disabled"),
    Output("patient-select", "value"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_patient_dropdown(item, submit):
    options = []
    disabled = True

    # triggered_id = dash.callback_context.triggered[0]['prop_id']
    # if triggered_id == 'next-btn.n_clicks':
    #     return

    if item:
        table = df_events.query(f'itemid == {item}')
        if table.empty:
            return options, disabled, None
        df_temp = pd.to_numeric(table['value'], errors='coerce')
        if df_temp.isna().sum().sum() / df_temp.shape[0] > 0.5:
            return options, disabled, None
        options = query_patients(item)
        disabled = False
        if options:
            first_patient = options[0]["value"]
            return options, disabled, first_patient
        return options, disabled, None
    return options, disabled, None


@app.callback(
    Output("tabs", "value"),
    [
        Input("patient-select", "value"),
        Input("submit-btn", "n_clicks"),
    ]
)
def update_tabs_view(patient, submit_n_clicks):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return "home-tab"
    elif patient is not None:
        raise PreventUpdate
    else:
        return "home-tab"


@app.callback(
    Output("all_patients_graph", "figure"),
    Output("blood_gas_graph", "figure"),
    Output("blood_gas_tab", "disabled"),
    Output("chemistry_graph", "figure"),
    Output("chemistry_tab", "disabled"),
    Output("cbc_graph", "figure"),
    Output("cbc_tab", "disabled"),
    [
        Input("item-select", "value"),
        Input("patient-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_graph(item, patient, submit):
    disabled = True

    if item is None:
        return {}, {}, disabled, {}, disabled, {}, disabled

    if patient:
        list_of_items = [bg_pair[0], bg_pair[1], chem_pair[0], chem_pair[1], cbc_pair[0], cbc_pair[1]]
        item_exists = []
        tabs = []
        for each_item in list_of_items:
            table = df_events.query(f'itemid == {each_item}')
            patients_with_item = set(table['subject_id'].unique())
            if patient in patients_with_item:
                item_exists.append(True)
            else:
                item_exists.append(False)
        for i in range(0, len(item_exists), 2):
            if item_exists[i] and item_exists[i + 1]:
                tabs.append(True)
            else:
                tabs.append(False)
        disabled = False
        if tabs[0]:
            tab_bg = (generate_tab_graph(item, patient, bg_pair, config=config.graphs.kwargs), disabled)
        else:
            tab_bg = ({}, True)
        if tabs[1]:
            tab_chem = (generate_tab_graph(item, patient, chem_pair, config=config.graphs.kwargs), disabled)
        else:
            tab_chem = ({}, True)
        if tabs[2]:
            tab_cbc = (generate_tab_graph(item, patient, cbc_pair, config=config.graphs.kwargs), disabled)
        else:
            tab_cbc = ({}, True)
        tab_item = generate_all_patients_graph(item, config=config.graphs.kwargs)

        return tab_item, tab_bg[0], tab_bg[1], tab_chem[0], tab_chem[1], tab_cbc[0], tab_cbc[1]
    return generate_all_patients_graph(item,
                                       config=config.graphs.kwargs), {}, disabled, {}, disabled, {}, disabled


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
    Output("related-datatable", "page_current"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("ontology-datatable", "active_cell"),
    ]
)
def reset_related_datatable_page(annotation, _, __, ___):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return 0
    elif triggered_id == 'skip-btn.n_clicks':
        return 0
    elif triggered_id == 'ontology-datatable.active_cell':
        return 0
    if not annotation:
        return 0


@app.callback(
    Output("ontology-results-outer", "hidden"),
    Output("ontology-datatable", "data"),
    Output("ontology-datatable", "columns"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("related-datatable", "active_cell"),
    ],
    [
        State("related-datatable", "data"),
    ]
)
def update_ontology_datatable(annotation, submit, related, curr_data):
    triggered_ids = dash.callback_context.triggered
    if triggered_ids[0]['prop_id'] == 'submit-btn.n_clicks' or not annotation:
        return True, None, []
    df_data = df_ontology.loc[df_ontology['LOINC_NUM'] == annotation]
    data = df_data.to_dict('records')
    columns = [{"name": i, "id": i} for i in df_data.columns]
    if related:
        df_data = pd.concat([df_data, df_ontology.loc[
            df_ontology['LOINC_NUM'] == [each_key for each_key in curr_data if each_key['id'] == related['row_id']][0][
                'CODE']]])
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

    query = ontology_dict[annotation]
    choices = list(df_ontology_new['LABEL'])
    if config.ontology.related.scorer == 'partial_ratio':
        df_data = generateRelatedOntologies(query, choices, method='partial_ratio', df_ontology=df_ontology_new)
    elif config.ontology.related.scorer == 'jaro_winkler':
        df_data = generateRelatedOntologies(query, choices, method='jaro_winkler', df_ontology=df_ontology_new)
    elif config.ontology.related.scorer == 'tf_idf':
        # NLP: tf_idf
        df_data = generateRelatedOntologies(query, choices, method='tf_idf',
                                            df_ontology=df_ontology_new,
                                            vectorizer=vectorizer,
                                            tf_idf_matrix=tf_idf_matrix)
    else:
        raise ScorerNotAvailable("Please define scorer from available options in the configuration file.")

    scores = df_data.iloc[:, 3]

    df_data.loc[(df_data.iloc[:, 3] >= np.percentile(scores, 66)), 'RELEVANCE'] = '🟢'
    df_data.loc[(df_data.iloc[:, 3] >= np.percentile(scores, 33)) &
                (df_data.iloc[:, 3] < np.percentile(scores, 66)), 'RELEVANCE'] = '🟡'
    df_data.loc[(df_data.iloc[:, 3] < np.percentile(scores, 33)), 'RELEVANCE'] = '🟠'

    data = df_data.to_dict('records')
    return False, data, True, True


@app.callback(
    Output("hidden-div", "children"),
    Output("refresh-url", "href"),
    # Output("df_items-store", "data"),
    # Output("df_events-store", "data"),
    # Output("df_ontology-store", "data"),
    # Output("df_ontology_new-store", "data"),
    # Output("itemsid_dict-store", "data"),
    # Output("ontology_dict-store", "data"),
    [
        Input("upload-data-btn", "contents"),
    ],
    [
        State("upload-data-btn", "filename"),
        State("upload-data-btn", "last_modified"),
    ],
    prevent_initial_call=True,
)
def update_config(contents, filename, last_modified):
    print("\n---------------------")
    print("UPDATING DASHBOARD...")
    print("---------------------\n")
    global config
    global df_items, df_events, df_ontology, df_ontology_new
    global itemsid_dict, ontology_dict
    global PATH_data, PATH_items, PATH_results, PATH_ontology, PATH_related
    global ngrams, vectorizer, tf_idf_matrix
    global annotated_list, skipped_list, unannotated_list
    global pairs, bg_pair, chem_pair, cbc_pair
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'yaml' in filename:
            config = yaml.unsafe_load(decoded)
        else:
            raise ConfigurationFileError

        PATH_data = config.directories.data.location
        PATH_items = config.directories.concepts.location
        PATH_results = config.directories.results
        PATH_ontology = config.ontology.location
        PATH_related = config.ontology.related.location

        if 'demo-data' in PATH_data:
            print("Demo data selected.")

        if config.ontology.related.scorer == 'tf_idf':
            with shelve.open(os.path.join(PATH_related, 'tf_idf.shlv'), protocol=5) as shlv:
                ngrams = shlv['ngrams']
                vectorizer = shlv['model']
                tf_idf_matrix = shlv['tf_idf_matrix']

        df_events = load_data(os.path.join(PATH_data, config.directories.data.filename))

        df_items, itemsid_dict = load_items(os.path.join(PATH_items, config.directories.concepts.filename))
        print("Data loaded.\n")

        df_ontology, ontology_dict = load_ontology(ontology=config.ontology.name)
        df_ontology_new = pd.DataFrame(
            {'CODE': list(ontology_dict.keys()), 'LABEL': list(ontology_dict.values())})
        df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})
        print('Ontology loaded.\n')

        annotated_list, skipped_list = load_annotations(PATH_results)
        unannotated_list = list(set(itemsid_dict.keys()) - set(annotated_list))
        unannotated_list.sort()

        # define item pairs for patient specific tabs
        pairs = []
        for each in config.graphs.pairs.values():
            pairs.append((each['label'], each['item_1'], each['item_2']))
        bg_pair = pairs[0][1:]
        chem_pair = pairs[1][1:]
        cbc_pair = pairs[2][1:]

    else:
        raise ConfigurationFileError

    return None, "/"  # , df_items.to_json(), json.dumps(itemsid_dict)


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
# PAGE LAYOUT #
######################################################################################################
def serve_layout():
    return html.Div(
        id="app-container",
        children=[
            dcc.Location(id='refresh-url', refresh=True),
            html.Div(id='hidden-div', hidden=True),
            # dcc.Store(id='df_items-store'),
            # dcc.Store(id='df_events-store'),
            # dcc.Store(id='df_ontology-store'),
            # dcc.Store(id='df_ontology_new-store'),
            # dcc.Store(id='itemsid_dict-store'),
            # dcc.Store(id='ontology_dict-store'),
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
                children=[
                    html.Img(src=app.get_asset_url("mimic.png"), style={'height': '120%', 'width': '10%'}),
                    html.H5("Welcome to the MIMIC-IV Clinical Annotation Dashboard"),
                    dcc.Upload(
                        id='upload-data-btn',
                        children=html.Button(
                            id='upload-btn',
                            children=[html.Img(src='assets/upload.png')],
                            style={'border-width': '0px'}
                        ),
                    ),
                ]
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
                                dcc.Tab(label=pairs[0][0], id="blood_gas_tab",
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
                                dcc.Tab(label=pairs[1][0], id="chemistry_tab",
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
                                dcc.Tab(label=pairs[2][0], id="cbc_tab",
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
                            html.Div(id='ontology-results-outer',
                                     hidden=True,
                                     children=[
                                         dcc.Clipboard(
                                             id='ontology-copy',
                                             title="Copy ontology Results",
                                             style={
                                                 "color": "#c9ddee",
                                                 "fontSize": 15,
                                                 "verticalAlign": "center",
                                                 'float': 'right',
                                                 'margin': 'auto'
                                             },
                                         ),
                                         html.P(children=[
                                             html.B('Compare Results (click to swap search terms):'),
                                         ]),
                                         html.Div(
                                             id="ontology-datatable-outer",
                                             className='ontology-datatable',
                                             hidden=False,
                                             children=[
                                                 dash_table.DataTable(id='ontology-datatable',
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
                                                                                  'state': 'active'
                                                                                  # 'active' | 'selected'
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


app.layout = serve_layout
######################################################################################################


# run
if __name__ == "__main__":
    app.run_server(port=8888, debug=True)
