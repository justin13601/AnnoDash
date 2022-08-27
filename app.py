#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""

import os
import io
import re
import sys
import csv
import time
import json
import yaml
import errno
import base64
import requests
from datetime import timedelta, datetime as dt
from collections import defaultdict
from ml_collections import config_dict
from zipfile import ZipFile

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import Dash, html, dcc, dash_table, ALL, ctx
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate
from dash.dash import no_update
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
import sqlite3

from related_ontologies.related import ngrams, generateRelatedOntologies, TfidfVectorizer, cosine_similarity


# from callbacks.all_callbacks import callback_manager


def big_query(query):
    client = bigquery.Client()
    query_job = client.query(query)  # API request
    print("The query data:")
    sys.stdout.flush()
    for row in query_job:
        # row values can be accessed by field name or index
        print("name={}, count={}".format(row[0], row["total_people"]))
    sys.stdout.flush()
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


class InvalidOntology(Exception):
    pass


def load_config(file):
    print(f'Loading {file}...')
    sys.stdout.flush()
    with open(file, "r") as f:
        configurations = yaml.unsafe_load(f)
        print('Done.\n')
        sys.stdout.flush()
        return configurations


config_file = 'config-demo.yaml'
if os.path.exists(config_file):
    print('Configuration file found.')
    sys.stdout.flush()
    config = load_config(config_file)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

# paths
PATH_data = config.directories.data.location
PATH_items = config.directories.concepts.location
PATH_results = config.directories.results
PATH_ontology = config.ontology.location
PATH_related = config.ontology.related.location

if not os.path.exists(PATH_results):
    os.makedirs(PATH_results)


def load_items(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    sys.stdout.flush()
    items = pd.read_csv(path)
    dictionary = pd.Series(items[items.columns[1]].values, index=items[items.columns[0]].values).to_dict()
    print('Done.\n')
    sys.stdout.flush()
    return items, dictionary


def tryConvertDate(dates):
    try:
        return dt.strptime(dates, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return dates


def load_data(path):
    filename = os.path.basename(path).strip()
    print(f'Loading {filename}...')
    sys.stdout.flush()
    data = pd.read_csv(path)
    # Date, format charttime
    data["charttime"] = data["charttime"].apply(
        lambda x: tryConvertDate(x)
    )  # String -> Datetime
    print('Done.\n')
    sys.stdout.flush()
    return data


# load data
def load_ontologies():
    print(f'Loading available ontologies:')
    sys.stdout.flush()
    ontology_files = [each_file.replace(".db", "") for each_file in os.listdir(PATH_ontology) if
                      each_file.endswith('.db')]
    if not ontology_files:
        raise InvalidOntology
    print('Done.\n')
    sys.stdout.flush()
    return ontology_files


def select_ontology(ontology):
    global df_ontology, ontology_dict, df_ontology_new
    database_file = f'{ontology}.db'
    path = os.path.join(PATH_ontology, database_file)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    if ontology not in list_of_ontologies:
        raise InvalidOntology
    df_ontology = pd.read_sql(f"SELECT * FROM {ontology}", conn)
    ontology_dict = pd.Series(df_ontology.LABEL.values, index=df_ontology.CODE.values).to_dict()
    df_ontology_new = pd.DataFrame(
        {'CODE': list(ontology_dict.keys()), 'LABEL': list(ontology_dict.values())})
    df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})
    print(f"{ontology.upper()} codes processed and selected.\n")
    sys.stdout.flush()
    return df_ontology, ontology_dict


def load_annotations(path):
    annotation_files = [each_json for each_json in os.listdir(path) if each_json.endswith('.json')]
    annotated = [int(each_item.strip('.json')) for each_item in annotation_files if
                 int(each_item.strip('.json')) in list(itemsid_dict.keys())]
    skipped = []
    for each_file in annotation_files:
        if int(each_file.strip('.json')) in list(itemsid_dict.keys()):
            with open(os.path.join(path, each_file)) as jsonFile:
                data = json.load(jsonFile)
                if type(data['annotatedid']) is not list:
                    if 'skipped' in data['annotatedid'].lower().strip():
                        skipped.append(int(each_file.strip('.json')))
    return annotated, skipped


def download_annotation(annotation):
    results_folder = config.directories.results
    path = os.path.join(results_folder, f"{annotation}.json")
    return send_file(path, as_attachment=True)


if 'demo-data' in PATH_data:
    print("Demo data selected.")
    sys.stdout.flush()

df_events = load_data(os.path.join(PATH_data, config.directories.data.filename))

df_items, itemsid_dict = load_items(os.path.join(PATH_items, config.directories.concepts.filename))
print("Data ready.\n")
sys.stdout.flush()

list_of_ontologies = load_ontologies()
df_ontology, ontology_dict = select_ontology(list_of_ontologies[0])
df_ontology_new = pd.DataFrame(
    {'CODE': list(ontology_dict.keys()), 'LABEL': list(ontology_dict.values())})
df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})
print('Ontology ready.\n')
sys.stdout.flush()

# load tf_idf matrix if LoincClassType_1 is a loaded ontology, can add other class types as long as it's fitted:
if 'LoincClassType_1' in list_of_ontologies:
    try:
        with shelve.open(os.path.join(PATH_related, 'tf_idf.shlv'), protocol=5) as shlv:
            ngrams = shlv['ngrams']
            vectorizer = shlv['model']
            tf_idf_matrix = shlv['tf_idf_matrix']
    except FileNotFoundError:
        print("Vectorizer shelve files not found, TF-IDF will not be available for LoincClassType_1.")
        sys.stdout.flush()

annotated_list, skipped_list = load_annotations(PATH_results)
unannotated_list = list(set(itemsid_dict.keys()) - set(annotated_list))
unannotated_list.sort()

# define item ref_vals for patient specific tabs
ref_vals = []
for each_ref_val in config.graphs.ref_vals.values():
    ref_group = [each_ref_val['label']]
    ref_group.extend([each_ref_val['items'][i] for i in range(len(each_ref_val['items']))])
    ref_vals.append(ref_group)
set_1 = ref_vals[0][1:]  # LABITEMS: 50821, 50818, add FiO2; ITEMS: 223900, 223901, 220739
set_2 = ref_vals[1][
        1:]  # LABITEMS: 50912, 50971, add Sodium & Glucose; ITEMS: 223834, 223835, 220339, add 223849, 229314
set_3 = ref_vals[2][1:]  # LABITEMS: 51222, 51300; ITEMS: 220045, 220181, 220210, 220277, 223761


######################################################################################################
# FUNCTIONS #
######################################################################################################
def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0

    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'

    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
          isinstance(df_column.dtype, pd.BooleanDtype) or
          isinstance(df_column.dtype, pd.CategoricalDtype) or
          isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
          isinstance(df_column.dtype, pd.IntervalDtype) or
          isinstance(df_column.dtype, pd.Int8Dtype) or
          isinstance(df_column.dtype, pd.Int16Dtype) or
          isinstance(df_column.dtype, pd.Int32Dtype) or
          isinstance(df_column.dtype, pd.Int64Dtype)):
        return 'numeric'
    else:
        return 'any'


def generate_ontology_options():
    ontology_options = [{"label": "LOINC® Core Edition (2.72)", "value": "loinc"},
                        {"label": "SNOMED-CT International Edition (07/31/2022)", "value": "snomed"}]
    return ontology_options


def generate_scorer_options(ontology):
    if ontology == 'LoincClassType_1':
        options = ["tf_idf", "jaro_winkler", "partial_ratio"]
    elif 'LoincClassType' in ontology:
        options = ["jaro_winkler", "partial_ratio"]
    elif 'SNOMED' in ontology:
        options = ["jaro_winkler", "partial_ratio", "UMLS"]
    else:
        options = ["fts5"]

    scorer_options = [{"label": each_scorer.replace('_', ' '), "value": each_scorer} for each_scorer in options]
    return scorer_options


def generate_all_patients_graph(item, **kwargs):
    table = df_events.query(f'itemid == {item}')
    if table.empty:
        return {}
    df_temp = pd.to_numeric(table['value'], errors='coerce')
    if df_temp.isna().sum().sum() / df_temp.shape[0] > 0.5:  # text data
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
        row_dict = df_items.query(f'itemid == {item}').iloc[:, 2:4].to_dict(orient='records')[0]
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor='rgba(0,0,0,0)'),
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
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    table_item_target = df_events.query(f'itemid == {item}')
    if not pd.to_numeric(table_item_target['value'], errors='coerce').isnull().all():
        table_item_target['value'] = pd.to_numeric(table_item_target['value'], errors='coerce')

    table_item_patient_target = table_item_target.query(f'subject_id == {patient}')
    units_target = list(table_item_target['valueuom'])[0]

    if table_item_patient_target.empty:
        return {}

    start_date = min(min(table_item_patient_target['charttime']),
                     min(table_item_patient_target['charttime'])) - timedelta(hours=12)
    end_date = start_date + timedelta(hours=96) + timedelta(hours=12)

    mask_plot = (table_item_patient_target['charttime'] > start_date) & (
            table_item_patient_target['charttime'] <= end_date)
    table_item_patient_target = table_item_patient_target.loc[mask_plot]
    table_item_patient_target = table_item_patient_target.sort_values(by="charttime")

    fig.add_trace(
        go.Scatter(x=table_item_patient_target["charttime"], y=table_item_patient_target["value"], mode='lines+markers',
                   name=f"{itemsid_dict[item]} ({units_target})", hovertemplate='%{y}'),
        secondary_y=True,
    )

    table_items = []
    table_item_patients = []
    units = []
    for each_item in template_items:
        if each_item == item:
            continue
        table_item = df_events.query(f'itemid == {each_item}')
        table_items.append(table_item)

        if not pd.to_numeric(table_item['value'], errors='coerce').isnull().all():
            table_item['value'] = pd.to_numeric(table_item['value'], errors='coerce')
        else:
            try:
                table_item['value'] = table_item['valuenum']
            except:
                pass

        table_item_patient = table_item.query(f'subject_id == {patient}')
        table_item_patients.append(table_item_patient)

        unit = list(table_item['valueuom'])[0]
        units.append(unit)

    empty = [table_item_patient.empty for table_item_patient in table_item_patients]
    if all(empty):
        return {}

    def series_names_gen(each_item):
        try:
            return itemsid_dict[each_item]
        except:
            return each_item

    series_names = [series_names_gen(each_item) for each_item in template_items if each_item != item]

    for i in range(len(table_item_patients)):
        mask = (table_item_patients[i]['charttime'] > start_date) & (table_item_patients[i]['charttime'] <= end_date)
        table_item_patients[i] = table_item_patients[i].loc[mask]
        table_item_patients[i] = table_item_patients[i].sort_values(by="charttime")

        fig.add_trace(
            go.Scatter(x=table_item_patients[i]["charttime"], y=table_item_patients[i]["value"], mode='lines+markers',
                       marker_symbol='circle-open', opacity=kwargs['config'].opacity,
                       name=f"{series_names[i]} ({units[i]})", hovertemplate='%{y}'),
            secondary_y=False,
        )

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
        # title={
        #     'text': f"{itemsid_dict[item]}",
        #     'y': 0.95,
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top',
        #     'font': dict(
        #         family=kwargs['config'].title_font,
        #         size=kwargs['config'].title_size,
        #         color=kwargs['config'].title_color
        #     )},
        xaxis_title="Time (Hours)",
        font=dict(
            family=kwargs['config'].text_font,
            size=kwargs['config'].text_size,
            color=kwargs['config'].text_color
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.47, bgcolor='rgba(0,0,0,0)'),
        legend_tracegroupgap=1,
        margin=dict(l=50, r=0, t=40, b=20),
        height=kwargs['config'].height
    )
    return fig


def query_patients(item):
    list_of_items = [item]
    list_of_items = list_of_items + set_1 + set_2 + set_3
    list_of_patient_sets = []
    target_table = df_events.query(f'itemid == {item}')
    if target_table.empty:
        return []
    for each_item in list_of_items:
        table = df_events.query(f'itemid == {each_item}')
        list_of_patient_sets.append(set(table['subject_id'].unique()))

    map_reduce_list = []
    for i in range(0, len(list_of_patient_sets)):
        for each_item in list_of_patient_sets[i]:
            map_reduce_list.append([each_item, i])

    dict_grouped = {}
    for each_ref in map_reduce_list:
        if each_ref[0] not in dict_grouped:
            dict_grouped[each_ref[0]] = each_ref[1:]
        else:
            dict_grouped[each_ref[0]].append(each_ref[1])

    second_map_reduce_list = []
    for each_key in dict_grouped:
        if len(dict_grouped[each_key]) > 1:
            second_map_reduce_list.append([tuple(dict_grouped[each_key]), each_key])

    ranked_patients = sorted(second_map_reduce_list, key=lambda x: (len(x[0]), x[1]))[::-1]
    ranked_patients = [patient for patient in ranked_patients if 0 in patient[0]]
    list_of_patients_ranked = [subject_id[1] for subject_id in ranked_patients]

    patients = [{"label": f"Patient {each_patient}", "value": each_patient} for each_patient in
                list_of_patients_ranked]
    return patients


def initialize_download_button(annotated):
    if annotated:
        return False
    return True


def initialize_all_patients_graph():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    fig = generate_all_patients_graph(items[0]["value"], config=config.graphs.kwargs)
    return fig


def initialize_tab_graph(template_items):
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(items[0]["value"])
    try:
        if not patients[0]:
            return {}
    except IndexError:
        return {}
    first_patient = patients[0]["value"]
    fig = generate_tab_graph(items[0]["value"], first_patient, template_items=template_items,
                             config=config.graphs.kwargs)
    return fig


def initialize_tab():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    patients = query_patients(items[0]["value"])
    try:
        if not patients[0]:
            return True
    except IndexError:
        return True
    first_patient = patients[0]["value"]
    disabled = update_graph(items[0]["value"], first_patient, None)[2]
    return disabled


def initialize_item_select():
    options = [
        {"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} if each_id in unannotated_list else
        {"label": f'{each_id}: {itemsid_dict[each_id]} ✔', "value": each_id} if each_id in list(
            set(annotated_list) - set(skipped_list)) else
        {"label": f'{each_id}: {itemsid_dict[each_id]} ⚠', "value": each_id} for each_id in itemsid_dict
    ]
    first_item = [each_id for each_id in list(itemsid_dict.keys()) if each_id in unannotated_list][0]
    return options, first_item


def initialize_patient_select():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in itemsid_dict]
    options = query_patients(items[0]["value"])
    if not options:
        return [], None
    first_patient = options[0]["value"]
    return options, first_patient


def annotate(item, annotation, ontology, skipped=False):
    if skipped:
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     'ontology': ontology,
                     'annotatedid': f'Skipped: {annotation}',
                     'annotatedlabel': 'n/a'
                     }
    else:
        # item_row = df_items.query(f'itemid == {item}')
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     'ontology': ontology,
                     # 'mimic_loinc': item_row['loinc_code'].item(),      # not available in mimic-iv-v2.0, removed in d-items
                     'annotatedid': [each_id['CODE'] for each_id in annotation],
                     'annotatedlabel': [each_id['LABEL'] for each_id in annotation]
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
    Output("ontology-filter-copy", "content"),
    [
        Input("ontology-filter-copy", "n_clicks"),
    ],
    [
        State("ontology-select", "value"),
    ]
)
def copy_ontology(_, ontology):
    if ontology is None:
        raise PreventUpdate
    clipboard = f'{ontology}'
    return clipboard


# @app.callback(
#     Output("patient-copy", "content"),
#     [
#         Input("patient-copy", "n_clicks"),
#     ],
#     [
#         State("patient-select", "value"),
#     ]
# )
# def copy_patient(_, patient):
#     if patient is None:
#         raise PreventUpdate
#     return str(patient)
#

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
    Output("submit-btn", "disabled"),
    [
        Input("ontology-datatable", "data"),
    ]
)
def enable_submit_button(ontology):
    if ontology:
        return False
    return True


@app.callback(
    Output("download-outer", "hidden"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def enable_download_button(_, __):
    return False


@app.callback(
    Output("skip-other-text", "disabled"),
    [
        Input("skip-radio-items", "value"),
    ],
    prevent_initial_call=True,
)
def show_skip_other(value):
    if value == 'Other':
        return False
    return True


@app.callback(
    Output("skip-other-text", "value"),
    Output("skip-radio-items", "value"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("skip-radio-items", "value"),
    ],
    prevent_initial_call=True,
)
def clear_reset_skip(_, __, skip):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return '', "Unsure"
    elif triggered_id == 'skip-btn.n_clicks':
        return '', "Unsure"
    elif skip != "Other":
        return '', no_update
    else:
        raise PreventUpdate


@app.callback(
    Output("download-annotations", "data"),
    [
        Input("download-btn", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def download_annotations(_):
    def write_archive(bytes_io):
        with ZipFile(bytes_io, mode="w") as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(config.directories.results):
                for filename in filenames:
                    # Add file to zip
                    zipObj.writestr(filename, os.path.basename(filename))

    return dcc.send_bytes(write_archive, "annotations.zip")


@app.callback(
    Output("related-datatable", "active_cell"),
    Output("related-datatable", "selected_cells"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("ontology-select", "value"),
        Input("ontology-datatable", "data_previous"),
    ],
    [
        State("ontology-datatable", "data"),
        State("item-select", "value"),
        State("skip-radio-items", "value"),
        State("skip-other-text", "value"),
    ],
    prevent_initial_call=True
)
def reset_annotation(_, __, ontology, prev_ontology_data, curr_ontology_data, item,
                     skip_reason, other_reason):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        annotate(item, curr_ontology_data, ontology)
        return None, []
    elif triggered_id == 'skip-btn.n_clicks':
        if skip_reason == 'Other':
            annotate(item, other_reason, ontology, skipped=True)
        else:
            annotate(item, skip_reason, ontology, skipped=True)
        return None, []
    elif prev_ontology_data is not None:
        return None, []
    else:
        raise PreventUpdate


@app.callback(
    Output("scorer-select", "value"),
    Output("scorer-select", "options"),
    [
        Input("ontology-select", "value"),
    ]
)
def update_scorer_select(ontology):
    scorer_options = generate_scorer_options(ontology)
    return scorer_options[0]["value"], scorer_options


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
def update_item_dropdown(_, __, options, value):
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
def update_patient_dropdown(item, _):
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
def update_tabs_view(patient, _):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return "home-tab"
    elif patient is not None:
        raise PreventUpdate
    else:
        return "home-tab"


@app.callback(
    Output("all_patients_graph", "figure"),
    Output("set_1_graph", "figure"),
    Output("set_1_tab", "disabled"),
    Output("set_2_graph", "figure"),
    Output("set_2_tab", "disabled"),
    Output("set_3_graph", "figure"),
    Output("set_3_tab", "disabled"),
    [
        Input("item-select", "value"),
        Input("patient-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_graph(item, patient, _):
    disabled = True

    if item is None:
        return {}, {}, disabled, {}, disabled, {}, disabled

    if patient:
        list_of_items = [set_1[0], set_1[1], set_2[0], set_2[1], set_3[0], set_3[1]]
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
            tab_1 = (generate_tab_graph(item, patient, set_1, config=config.graphs.kwargs), disabled)
        else:
            tab_1 = ({}, True)
        if tabs[1]:
            tab_2 = (generate_tab_graph(item, patient, set_2, config=config.graphs.kwargs), disabled)
        else:
            tab_2 = ({}, True)
        if tabs[2]:
            tab_3 = (generate_tab_graph(item, patient, set_3, config=config.graphs.kwargs), disabled)
        else:
            tab_3 = ({}, True)
        tab_item = generate_all_patients_graph(item, config=config.graphs.kwargs)

        return tab_item, tab_1[0], tab_1[1], tab_2[0], tab_2[1], tab_3[0], tab_3[1]
    return generate_all_patients_graph(item,
                                       config=config.graphs.kwargs), {}, disabled, {}, disabled, {}, disabled


@app.callback(
    Output("related-datatable", "page_current"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("search-btn", "n_clicks"),
    ]
)
def reset_related_datatable_page(item, _, __, ___):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return 0
    elif triggered_id == 'skip-btn.n_clicks':
        return 0
    elif triggered_id == 'search-btn.n_clicks':
        return 0
    if not item:
        return 0


@app.callback(
    Output("ontology-datatable", "data"),
    Output("ontology-datatable", "columns"),
    # Output("ontology-datatable", "tooltip_data"),
    [
        Input("submit-btn", "n_clicks"),
        Input("related-datatable", "active_cell"),
    ],
    [
        State("related-datatable", "data"),
        State("ontology-datatable", "data"),
        State("ontology-datatable", "columns"),
    ]
)
def update_ontology_datatable(_, related, curr_data_related, curr_data_ontology, curr_ontology_cols):
    triggered_ids = dash.callback_context.triggered
    if triggered_ids[0]['prop_id'] == 'submit-btn.n_clicks':
        return curr_data_ontology[0:0], curr_ontology_cols#, []
    if not curr_data_ontology:
        df_data = pd.DataFrame(columns=df_ontology_new.columns)
    else:
        df_data = pd.DataFrame.from_records(curr_data_ontology)
    columns = [{"name": 'CODE', "id": 'CODE'}, {"name": 'LABEL', "id": 'LABEL'}]
    if related:
        if curr_data_related[related['row_id']]['CODE'] in [each_selected['CODE'] for each_selected in
                                                            curr_data_ontology]:
            print('Ontology code already added!')
            sys.stdout.flush()
            raise PreventUpdate
        print('Adding ontology code...')
        sys.stdout.flush()
        df_data = pd.concat(
            [df_data, df_ontology_new.loc[df_ontology_new['CODE'] == curr_data_related[related['row_id']]['CODE']]])

    def table_gen(each_row):
        try:
            df_tooltip = pd.DataFrame()
            for each_ontology in list_of_ontologies:
                database_file = f'{each_ontology}.db'
                path = os.path.join(PATH_ontology, database_file)
                conn = sqlite3.connect(path)
                c = conn.cursor()
                df_tooltip = pd.read_sql(
                    f"SELECT * FROM {each_ontology} WHERE CODE MATCH '\"{each_row['CODE']}\"'",
                    conn)
                if not df_tooltip.empty:
                    break
            return df_tooltip
        except:
            return pd.DataFrame()

    tooltip_tables = [table_gen(each_row) for each_row in df_data.to_dict('records')]

    def dict_gen(each_table):
        try:
            return each_table.to_dict('records')[0]
        except:
            return {}

    tooltip_dicts = [dict_gen(each_table) for each_table in tooltip_tables]
    tooltip_outputs = []
    for each_dict in tooltip_dicts:
        tooltip_output = {
            'value': "\n\n".join([f'**{each_item}**: {each_dict[each_item]}' for each_item in each_dict.keys()
                                  if each_item not in ['CODE', 'COMPONENT', 'LABEL']]),
            'type': 'markdown'}
        tooltip_outputs.append({'LABEL': tooltip_output})
    data = df_data.to_dict('records')
    return data, columns#, tooltip_outputs


@app.callback(
    Output("related-datatable", "data"),
    Output("related-datatable", "columns"),
    Output("before_loading", "hidden"),
    Output("after_loading", "hidden"),
    Output("related-datatable", "tooltip_data"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("scorer-select", "value"),
        Input("ontology-select", "value"),
        Input("search-btn", "n_clicks"),
    ],
    [
        State("search-input", "value"),
    ]
)
def update_related_datatable(item, _, scorer, ontology_filter, __, search_string):
    if not item:
        return None, [], False, False, []

    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'ontology-select.value':
        select_ontology(ontology_filter)

    query = itemsid_dict[item]
    choices = list(df_ontology_new['LABEL'])
    if scorer == 'partial_ratio':
        df_data = generateRelatedOntologies(query, choices, method='partial_ratio', df_ontology=df_ontology_new)
    elif scorer == 'jaro_winkler':
        df_data = generateRelatedOntologies(query, choices, method='jaro_winkler', df_ontology=df_ontology_new)
    elif scorer == 'tf_idf':
        # NLP: tf_idf
        df_data = generateRelatedOntologies(query, choices, method='tf_idf',
                                            df_ontology=df_ontology_new,
                                            vectorizer=vectorizer,
                                            tf_idf_matrix=tf_idf_matrix)
    elif scorer == 'UMLS':
        results = generateRelatedOntologies(query, choices, method='UMLS', apikey=config.ontology.related.umls_apikey)
        query_2 = itemsid_dict[item]
        results_2 = generateRelatedOntologies(query_2, choices, method='UMLS',
                                              apikey=config.ontology.related.umls_apikey)
        results_all = results + results_2
        df_data = pd.DataFrame.from_records(results_all, columns=['LABEL', 'CODE'])
        df_data[scorer] = np.array([100] * len(df_data['CODE']))
        df_data['id'] = np.arange(len(df_data['CODE']))
        col_id = df_data.pop('id')
        df_data.insert(0, col_id.name, col_id)
    elif triggered_id == 'search-btn.n_clicks':
        database_file = f'{ontology_filter}.db'
        path = os.path.join(PATH_ontology, database_file)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        df_data = pd.read_sql(f"SELECT * FROM {ontology_filter} WHERE LABEL MATCH '{search_string}' ORDER BY rank",
                              conn)
        if df_data.empty:
            return None, [], True, True, []
        match_scores = [round(100 / len(df_data['CODE']) * i) for i in range(len(df_data['CODE']))]
        match_scores = match_scores[::-1]
        df_data[scorer] = match_scores
        df_data['id'] = np.arange(len(df_data['CODE']))
        col_id = df_data.pop('id')
        df_data.insert(0, col_id.name, col_id)
    elif scorer == 'fts5':
        database_file = f'{ontology_filter}.db'
        path = os.path.join(PATH_ontology, database_file)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        tokens = re.split('\W+', itemsid_dict[item])
        search_term = ' OR '.join(tokens)
        df_data = pd.read_sql(f"SELECT * FROM {ontology_filter} WHERE LABEL MATCH '{search_term}' ORDER BY rank", conn)
        if df_data.empty:
            return None, [], True, True, []
        match_scores = [round(100 / len(df_data['CODE']) * i) for i in range(len(df_data['CODE']))]
        match_scores = match_scores[::-1]
        df_data[scorer] = match_scores
        df_data['id'] = np.arange(len(df_data['CODE']))
        col_id = df_data.pop('id')
        df_data.insert(0, col_id.name, col_id)
    else:
        raise ScorerNotAvailable

    scores = df_data[scorer]

    df_data.loc[(df_data[scorer] >= np.percentile(scores, 66)), 'RELEVANCE'] = '🟢'
    df_data.loc[(df_data[scorer] >= np.percentile(scores, 33)) &
                (df_data[scorer] < np.percentile(scores, 66)), 'RELEVANCE'] = '🟡'
    df_data.loc[(df_data[scorer] < np.percentile(scores, 33)), 'RELEVANCE'] = '🟠'

    data = df_data.to_dict('records')
    columns = list(df_data.columns)
    columns.remove(scorer)
    columns.remove('id')
    columns.remove('RELEVANCE')
    columns.insert(0, 'RELEVANCE')
    return_columns = [
        {'name': each_col, 'id': each_col, 'type': table_type(df_data[each_col])} if each_col != 'RELEVANCE'
        else {'name': '', 'id': each_col} for each_col in columns]

    tooltip_dict = df_data[scorer].to_dict()
    tooltip_outputs = []
    for each_value in tooltip_dict:
        tooltip_output = {
            'value': f'**{tooltip_dict[each_value]}**',
            'type': 'markdown'}
        tooltip_outputs.append({'RELEVANCE': tooltip_output})
    return data, return_columns, True, True, tooltip_outputs


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
    sys.stdout.flush()
    print("UPDATING DASHBOARD...")
    sys.stdout.flush()
    print("---------------------\n")
    sys.stdout.flush()
    global config
    global list_of_ontologies
    global df_items, df_events, df_ontology, df_ontology_new
    global itemsid_dict, ontology_dict
    global PATH_data, PATH_items, PATH_results, PATH_ontology, PATH_related
    global ngrams, vectorizer, tf_idf_matrix
    global annotated_list, skipped_list, unannotated_list
    global ref_vals, set_1, set_2, set_3
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

        if not os.path.exists(PATH_results):
            os.makedirs(PATH_results)

        if 'demo-data' in PATH_data:
            print("Demo data selected.")
            sys.stdout.flush()

        # load tf_idf matrix if LoincClassType_1 is a loaded ontology, can add other class types as long as it's fitted:
        if 'LoincClassType_1' in list_of_ontologies:
            try:
                with shelve.open(os.path.join(PATH_related, 'tf_idf.shlv'), protocol=5) as shlv:
                    ngrams = shlv['ngrams']
                    vectorizer = shlv['model']
                    tf_idf_matrix = shlv['tf_idf_matrix']
            except FileNotFoundError:
                print("Vectorizer shelve files not found, TF-IDF will not be available for LoincClassType_1.")
                sys.stdout.flush()

        df_events = load_data(os.path.join(PATH_data, config.directories.data.filename))

        df_items, itemsid_dict = load_items(os.path.join(PATH_items, config.directories.concepts.filename))
        print("Data loaded.\n")
        sys.stdout.flush()

        list_of_ontologies = load_ontologies()
        df_ontology, ontology_dict = select_ontology(list_of_ontologies[0])
        df_ontology_new = pd.DataFrame(
            {'CODE': list(ontology_dict.keys()), 'LABEL': list(ontology_dict.values())})
        df_ontology_new = df_ontology_new.reset_index().rename(columns={"index": "id"})
        print('Ontology loaded.\n')
        sys.stdout.flush()

        annotated_list, skipped_list = load_annotations(PATH_results)
        unannotated_list = list(set(itemsid_dict.keys()) - set(annotated_list))
        unannotated_list.sort()

        # define item ref_vals for patient specific tabs
        ref_vals = []
        for each_ref_val in config.graphs.ref_vals.values():
            ref_group = [each_ref_val['label']]
            ref_group.extend([each_ref_val['items'][i] for i in range(len(each_ref_val['items']))])
            ref_vals.append(ref_group)
        set_1 = ref_vals[0][1:]
        set_2 = ref_vals[1][1:]
        set_3 = ref_vals[2][1:]

    else:
        raise ConfigurationFileError

    return None, "/"  # , df_items.to_json(), json.dumps(itemsid_dict)


@app.callback(
    Output("metadata-tooltip", "children"),
    [
        Input("item-select", "value"),
    ],
)
def update_metadata_tooltip(annotation):
    metadata_table = df_items.query(f"itemid == {annotation}")
    metadata_table = metadata_table.dropna(axis=1)
    metadata = metadata_table.to_dict('records')[0]
    output = [html.P(f"{each_item}: {metadata[each_item]}") for each_item in metadata.keys() if
              each_item not in ['itemid', 'label']]
    return output


@app.callback(
    Output("search-input", "placeholder"),
    [
        Input("ontology-select", "value"),
    ],
)
def update_search_placeholder(ontology):
    placeholder = ''
    if ontology == 'loinc':
        placeholder = 'Search all LOINC®...'
    elif ontology == 'snomed':
        placeholder = 'Search all SNOMED-CT...'
    return placeholder


######################################################################################################
# PAGE LAYOUT #
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
            dbc.Offcanvas(
                children=[
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.P("This is the content of the first section"),
                                ],
                                title="About",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P("This is the content of the second section"),
                                ],
                                title="Guide",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P("This is the content of the third section"),
                                ],
                                title="Acknowledgements",
                            ),
                        ],
                    ),
                ],
                id="offcanvas",
                title="MIMIC-Dash",
                is_open=False,
                placement='end',
                style={'color': 'black'},
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
                            'margin-top': '0px'
                        },
                    )
                ]
            ),
            html.P("Select Source Concept:"),
            dcc.Dropdown(
                id="item-select",
                clearable=False,
                value=initialize_item_select()[1],
                style={"border-radius": 0, "margin-bottom": "15px"},
                options=initialize_item_select()[0],
            ),
            html.Hr(
                style={
                    'margin-bottom': '13px'
                }
            ),
            html.Div(
                id='ontology-copy-outer',
                hidden=False,
                children=[
                    dcc.Clipboard(
                        id='ontology-filter-copy',
                        title="Copy Filtered Ontology",
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
            html.P("Filter Ontology:"),
            html.Div(
                children=[
                    dcc.Dropdown(
                        id="ontology-select",
                        value=generate_ontology_options()[0]['value'],
                        options=generate_ontology_options(),
                        disabled=False,
                        clearable=False,
                        style={"border-radius": 0},
                    ),
                ],
            ),
            html.Br(),
            html.Div(
                hidden=True,
                children=[
                    dcc.Dropdown(
                        id="scorer-select",
                        value=None,
                        options=[],
                        disabled=False,
                        clearable=False,
                        style={"border-radius": 0},
                    ),
                ],
            ),
            html.Div(
                style={
                    'margin-bottom': '-10px'
                }
            ),
            dcc.Clipboard(
                id='ontology-copy',
                title="Copy Ontology Codes",
                style={
                    "color": "#c9ddee",
                    "fontSize": 15,
                    "verticalAlign": "center",
                    'float': 'right',
                    'margin': 'auto'
                },
            ),
            html.P("Target Ontology Concepts:"),
            html.Div(id='ontology-results-outer',
                     hidden=False,
                     children=[
                         html.Div(
                             id="ontology-datatable-outer",
                             className='ontology-datatable-class',
                             hidden=False,
                             children=[
                                 dash_table.DataTable(id='ontology-datatable',
                                                      data=None,
                                                      columns=[{"name": 'CODE', "id": 'CODE'},
                                                               {"name": 'LABEL', "id": 'LABEL'}],
                                                      # fixed_rows={'headers': True},
                                                      tooltip_data=[],
                                                      css=[
                                                          {
                                                              'selector': '.dash-tooltip',
                                                              'rule': 'border-width: 0px;'
                                                          },
                                                          {
                                                              'selector': '.dash-table-tooltip',
                                                              'rule': 'background-color: #000000; color: #ffffff; padding: 10px 0px 5px 10px; border-radius: 5px; line-height: 15px;'
                                                          }
                                                      ],
                                                      style_data={
                                                          'whiteSpace': 'normal',
                                                          'height': 'auto',
                                                          'lineHeight': '15px',
                                                      },
                                                      style_table={
                                                          'height': '136px',
                                                          'overflowY': 'auto',
                                                          'backgroundColor': 'white'
                                                      },
                                                      style_cell={
                                                          'textAlign': 'left',
                                                          'backgroundColor': 'transparent',
                                                          'color': 'black',
                                                          'overflow': 'hidden',
                                                          'textOverflow': 'ellipsis',
                                                          'maxWidth': 0
                                                      },
                                                      style_header={
                                                          'fontWeight': 'bold',
                                                          'color': '#2c8cff'
                                                      },
                                                      style_data_conditional=[
                                                          {
                                                              'if': {'state': 'active'},
                                                              'backgroundColor': 'transparent',
                                                              'border': '1px solid lightgray'
                                                          },
                                                          {
                                                              'if': {'column_id': 'CODE'},
                                                              'width': '18%',
                                                              'minWidth': '18%',
                                                              'maxWidth': '18%',
                                                          },
                                                          {
                                                              'if': {'column_id': 'LABEL'},
                                                              'width': '80%',
                                                              'minWidth': '80%',
                                                              'maxWidth': '80%',
                                                          },
                                                      ],
                                                      row_deletable=True,
                                                      tooltip_delay=0,
                                                      tooltip_duration=None,
                                                      )
                             ]
                         )]
                     ),
            html.Div(
                id="submit-btn-outer",
                hidden=False,
                children=[
                    html.Button(id="submit-btn", children="Submit & Next", n_clicks=0,
                                style={'width': '100%', 'color': 'white',
                                       'margin-top': '12px', 'margin-bottom': '6px'},
                                disabled=True),
                ],
            ),
            html.Hr(
                style={
                    'margin-top': '14px',
                    'margin-bottom': '17px'
                }
            ),
            html.Div(
                id='search-ontology-outer',
                children=[
                    html.Div(id='search-outer',
                             hidden=False,
                             children=[
                                 dcc.Input(
                                     id="search-input",
                                     placeholder="",
                                     debounce=True,
                                     style={"width": '69%', 'margin-left': '0px', 'float': 'left'},
                                     autoFocus=True,
                                 ),
                             ]),
                    html.Button(id="search-btn", children="Search", n_clicks=0,
                                style={'width': '30%', 'color': 'white',
                                       'float': 'right'},
                                disabled=False),
                ],
                style={
                    'margin-top': '7px'
                },
            ),
        ],
        style={'width': '100%', 'color': 'black',
               'margin-top': '-20px'}
    )


def serve_layout():
    return html.Div(
        id="app-container",
        children=[
            dcc.Location(id='refresh-url', refresh=True),
            html.Div(id='hidden-div', hidden=True),
            dbc.Tooltip(
                id='metadata-tooltip',
                class_name='custom-tooltip',
                children=update_metadata_tooltip(initialize_item_select()[1]),
                target="item-select",
                placement='right',
                fade=True,
            ),
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
                    html.H5(""),
                    html.Div(children=[
                        # html.Div(
                        #     id='patient-copy-outer',
                        #     hidden=False,
                        #     children=[
                        #         dcc.Clipboard(
                        #             id='patient-copy',
                        #             title="Copy Patient ID",
                        #             style={
                        #                 "color": "#c9ddee",
                        #                 "fontSize": 15,
                        #                 "verticalAlign": "center",
                        #                 'float': 'right',
                        #                 'margin': 'auto'
                        #             },
                        #         )
                        #     ]
                        # ),
                        dcc.Dropdown(
                            id="patient-select",
                            value=initialize_patient_select()[1],
                            style={'position': 'relative', 'bottom': '2px', 'border-radius': '0px',
                                   'color': 'black'},
                            options=initialize_patient_select()[0],
                            disabled=False,
                        ),
                    ],
                        style={'width': '20%', 'margin-right': '5%'}),
                    dcc.Upload(
                        id='upload-data-btn',
                        children=[
                            html.Button(
                                id='upload-btn',
                                children=[html.Img(src='assets/upload.png', title="Upload config.yaml")],
                                style={'border-width': '0px'}
                            ),
                        ]
                    ),
                ]
            ),
            html.Div(
                id="columns-card",
                className='columns-card',
                children=[
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
                                        dcc.Tab(label='Distribution Overview\n(All Patients)', id="all_patients_tab",
                                                disabled=False,
                                                value='home-tab',
                                                style={'color': '#1a75f9',
                                                       'padding-top': '14px',
                                                       'white-space': 'pre'},
                                                selected_style={
                                                    'color': '#1a75f9',
                                                    'border-top-width': '3px',
                                                    'padding-top': '14px',
                                                    'white-space': 'pre',
                                                },
                                                disabled_style={
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                children=[
                                                    html.Div(
                                                        className='tab-outer',
                                                        children=[
                                                            dcc.Graph(
                                                                style={'height': '360px'},
                                                                id="all_patients_graph",
                                                                figure=initialize_all_patients_graph()
                                                            )
                                                        ]),
                                                ]),
                                        dcc.Tab(label=f"{ref_vals[0][0]}\n(Single Patient)", id="set_1_tab",
                                                disabled=initialize_tab(),
                                                style={'color': '#1a75f9',
                                                       'padding-top': '14px',
                                                       'white-space': 'pre'},
                                                selected_style={
                                                    'color': '#1a75f9',
                                                    'border-top-width': '3px',
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                disabled_style={
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        style={'height': '360px'},
                                                        id="set_1_graph",
                                                        figure=initialize_tab_graph(set_1)
                                                    )
                                                ]),
                                        dcc.Tab(label=f"{ref_vals[1][0]}\n(Single Patient)", id="set_2_tab",
                                                disabled=initialize_tab(),
                                                style={'color': '#1a75f9',
                                                       'padding-top': '14px',
                                                       'white-space': 'pre'},
                                                selected_style={
                                                    'color': '#1a75f9',
                                                    'border-top-width': '3px',
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                disabled_style={
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        style={'height': '360px'},
                                                        id="set_2_graph",
                                                        figure=initialize_tab_graph(set_2)
                                                    )
                                                ]),
                                        dcc.Tab(label=f"{ref_vals[2][0]}\n(Single Patient)", id="set_3_tab",
                                                disabled=initialize_tab(),
                                                style={'color': '#1a75f9',
                                                       'padding-top': '14px',
                                                       'white-space': 'pre'},
                                                selected_style={
                                                    'color': '#1a75f9',
                                                    'border-top-width': '3px',
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                disabled_style={
                                                    'padding-top': '14px',
                                                    'white-space': 'pre'
                                                },
                                                children=[
                                                    dcc.Graph(
                                                        style={'height': '360px'},
                                                        id="set_3_graph",
                                                        figure=initialize_tab_graph(set_3)
                                                    )
                                                ]),
                                    ], id='tabs', value='home-tab', style={'height': '75px'}),
                                ],
                            ),
                            html.Div(
                                id="skip-outer",
                                hidden=False,
                                style={'margin-top': '6px'},
                                children=[
                                    dcc.RadioItems(
                                        id='skip-radio-items',
                                        options=['Unsure', 'Invalid Source Data', 'Other'],
                                        value='Unsure',
                                        style={'width': '43%', 'color': 'white', 'textAlign': 'left',
                                               'verticalAlign': 'center', 'float': 'left', 'margin-top': '6px'},
                                        labelStyle={'margin-right': '30px'},
                                        inline=True,
                                    ),
                                    html.Div(id='skip-other-outer',
                                             hidden=False,
                                             children=[
                                                 dcc.Input(
                                                     id="skip-other-text",
                                                     placeholder="Reason...",
                                                     debounce=True,
                                                     style={"width": '24%', 'margin-left': '0px', 'float': 'left'},
                                                     autoFocus=True,
                                                     disabled=True
                                                 ),
                                             ]),
                                    html.Button(id="skip-btn", children="Skip", n_clicks=0,
                                                style={'width': '30%', 'color': 'white',
                                                       'float': 'right'},
                                                disabled=False),
                                ],
                            ),
                        ],
                    ),
                ]
            ),
            html.Div(
                id='results-card',
                className='results-card',
                children=[
                    html.Div(
                        id="annotation-outer",
                        hidden=False,
                        children=[
                            html.Div(id='before_loading', hidden=False),
                            dcc.Loading(
                                id="related-loading",
                                type="dot",
                                color='#2c89f2',
                                children=[
                                    dcc.Clipboard(
                                        id='related-copy',
                                        title="Copy Search Results",
                                        style={
                                            "color": "#c9ddee",
                                            "fontSize": 15,
                                            "verticalAlign": "center",
                                            'float': 'right',
                                            'margin': 'auto'
                                        },
                                    ),
                                    html.P(
                                        style={'margin-top': '15px'},
                                        children=[
                                            html.B('Results (click on rows to select):'),
                                        ]),
                                    html.Div(
                                        id="related-datatable-outer",
                                        className='related-datatable',
                                        hidden=False,
                                        children=dash_table.DataTable(id='related-datatable',
                                                                      data=None,
                                                                      columns=[],
                                                                      tooltip_data=[],
                                                                      sort_action='native',
                                                                      # fixed_rows={'headers': True},
                                                                      filter_action='native',
                                                                      filter_options={'case': 'insensitive'},
                                                                      style_data={
                                                                          'width': 'auto',
                                                                          'maxWidth': '100px',
                                                                          'minWidth': '100px',
                                                                          'whiteSpace': 'normal'
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
                                                                              'if': {'column_id': 'LABEL'},
                                                                              'width': '200px',
                                                                              'maxWidth': '200px',
                                                                              'minWidth': '200px',
                                                                          },
                                                                          {
                                                                              'if': {'column_id': 'RELEVANCE'},
                                                                              'width': '1px',
                                                                              'maxWidth': '1px',
                                                                              'minWidth': '1px',
                                                                          }
                                                                      ],
                                                                      page_size=20,
                                                                      # virtualization=True,
                                                                      merge_duplicate_headers=True,
                                                                      style_as_list_view=True,
                                                                      css=[
                                                                          {
                                                                              'selector': '.previous-page, .next-page, '
                                                                                          '.first-page, .last-page',
                                                                              'rule': 'color: #2c8cff'
                                                                          },
                                                                          {
                                                                              'selector': '.previous-page:hover',
                                                                              'rule': 'color: #002552'
                                                                          },
                                                                          {
                                                                              'selector': '.next-page:hover',
                                                                              'rule': 'color: #002552'
                                                                          },
                                                                          {
                                                                              'selector': '.first-page:hover',
                                                                              'rule': 'color: #002552'
                                                                          },
                                                                          {
                                                                              'selector': '.last-page:hover',
                                                                              'rule': 'color: #002552'
                                                                          },
                                                                          {
                                                                              'selector': '.column-header--sort:hover',
                                                                              'rule': 'color: #2c8cff'
                                                                          },
                                                                          {
                                                                              'selector': 'input.dash-filter--case--insensitive',
                                                                              'rule': 'border-color: #2c8cff !important; border-radius: 3px; border-style: solid; border-width: 2px; color: #2c8cff !important;'
                                                                          },
                                                                          {
                                                                              'selector': '.dash-tooltip',
                                                                              'rule': 'border-width: 0px; background-color: #000000;'
                                                                          },
                                                                          {
                                                                              'selector': '.dash-table-tooltip',
                                                                              'rule': 'background-color: #000000; color: #ffffff; border-radius: 5px; margin-top: 5px; line-height: 15px ; width: fit-content; max-width: 25px; min-width: unset;'
                                                                          }
                                                                      ],
                                                                      tooltip_delay=0,
                                                                      tooltip_duration=None,
                                                                      )
                                    ),
                                ]
                            ),
                            html.Div(id='after_loading', hidden=False),
                        ],
                    ),
                    html.Div(
                        id="download-outer",
                        hidden=initialize_download_button(annotated_list),
                        children=[
                            html.Br(),
                            html.Button(id="download-btn", children="Download current annotations.zip", n_clicks=0,
                                        style={'width': '100%', 'color': 'white'},
                                        disabled=False),
                            dcc.Download(id="download-annotations"),
                        ]
                    ),
                ]
            ),
        ],
    )


app.layout = serve_layout
######################################################################################################


# run app.py (MIMIC-Dash v2)
if __name__ == "__main__":
    app.run_server(port=8888, debug=True)
