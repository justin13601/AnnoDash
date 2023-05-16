import os
import re
# import io
import sys
# import csv
import time
import json

import yaml
import errno
import base64
# import requests
from datetime import timedelta, datetime as dt
from collections import Counter
# from collections import defaultdict
# from ml_collections import config_dict
from zipfile import ZipFile

# import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
import dash_bootstrap_components as dbc
# import dash_mantine_components as dmc
# from dash_iconify import DashIconify
from dash import Dash, html, dcc, dash_table, ALL, ctx
from dash.dependencies import State, Input, Output, ClientsideFunction
from dash.exceptions import PreventUpdate
from dash.dash import no_update
# from flask import Flask, send_file

import numpy as np
import pandas as pd
import sqlite3
from google.cloud import storage, bigquery
# import scipy
# import scipy.sparse as sp
# import lucene
# import jaro
# import pickle
# from fuzzywuzzy import fuzz, process
# from ftfy import fix_text

from related_ontologies.related import generateRelatedOntologies
from src.search import *
from src.rank import *
from src.app.app import app


######################################################################################################
# DATA & HELPER FUNCTIONS #
######################################################################################################

class AnnoDashError(Exception):
    pass


class ConfigurationFileError(Exception):
    pass


class InvalidOntology(Exception):
    pass


def load_config(file):
    print(f'Loading {file}...')
    with open(file, "r") as f:
        configurations = yaml.unsafe_load(f)
        print('Done.\n')
        return configurations


config_file = 'src/app/config.yaml'
if os.path.exists(config_file):
    print('Configuration file found.')
    config = load_config(config_file)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

# paths
PATH_data = config.data
PATH_items = config.concepts
PATH_results = config.results
PATH_ontology = config.ontology.location

if not os.path.exists(PATH_results):
    os.makedirs(PATH_results)


def load_items(item_path):
    filename = os.path.basename(item_path).strip()
    print(f'Loading {filename}...')
    items = pd.read_csv(item_path)
    dictionary = pd.Series(items[items.columns[1]].values, index=items[items.columns[0]].values).to_dict()
    print('Done.\n')
    return items, dictionary


def tryConvertDate(dates):
    try:
        return dt.strptime(dates, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return dates


def load_data(data_path):
    filename = os.path.basename(data_path).strip()
    print(f'Loading {filename}...')
    data = pd.read_csv(data_path)
    # Date, format charttime
    data["charttime"] = data["charttime"].apply(
        lambda x: tryConvertDate(x)
    )  # String -> Datetime
    print('Done.\n')
    return data


# load data
def list_available_ontologies():
    print(f'Loading available ontologies...')
    if '.appspot.com' in PATH_ontology:
        ontologies = ['loinc', 'snomed']
    else:
        directory_contents = os.listdir(PATH_ontology)
        ontologies = [item for item in directory_contents if os.path.isdir(os.path.join(PATH_ontology, item))]
    if not ontologies:
        raise InvalidOntology
    print(f"{', '.join(each_ontology.upper() for each_ontology in ontologies)} codes available.\n")
    return ontologies


def load_annotations(annotations_path):
    annotation_files = [each_json for each_json in os.listdir(annotations_path) if each_json.endswith('.json')]
    annotated = [int(each_item.strip('.json')) for each_item in annotation_files if
                 int(each_item.strip('.json')) in list(itemsid_dict.keys())]
    skipped = []
    for each_file in annotation_files:
        if int(each_file.strip('.json')) in list(itemsid_dict.keys()):
            with open(os.path.join(annotations_path, each_file)) as jsonFile:
                data = json.load(jsonFile)
                if type(data['annotatedid']) is not list:
                    if 'skipped' in data['annotatedid'].lower().strip():
                        skipped.append(int(each_file.strip('.json')))
    return annotated, skipped


if 'demo-data' in PATH_data:
    print("Demo data selected.")

df_events = load_data(PATH_data)

df_items, itemsid_dict = load_items(PATH_items)
print("Data ready.\n")

list_of_ontologies = list_available_ontologies()
print('Ontology ready.\n')

annotated_list, skipped_list = load_annotations(PATH_results)
unannotated_list = list(set(itemsid_dict.keys()) - set(annotated_list))
unannotated_list.sort()


def set_up(ontologies: list) -> (dict, dict):
    search_objects = {}
    for ontology in ontologies:
        database_file = f'{ontology}.db'
        database_path = os.path.join(os.path.join(PATH_ontology, ontology), database_file)

        search_objects[ontology] = SearchSQLite(ontology, database_path)

    try:
        index_objects = prepare_search(
            method=config.ontology.search,
            PATH_ontology=PATH_ontology,
            list_of_ontologies=list_of_ontologies
        )
    except:
        raise AnnoDashError
    return search_objects, index_objects


sql_searchers, my_indexes = set_up(list_of_ontologies)


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
    elif config.ontology.search == 'sqlite':
        options = ["fts5"]
    elif config.ontology.search == 'pylucene':
        options = ["pylucene"]
    elif config.ontology.search == 'elastic':
        options = ["elastic"]
    else:
        raise AnnoDashError

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
        fig.update_traces(marker_color='rgb(100,169,252)')
        ylabel = 'Count'
        del df_data
        del table
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
        del table
    del df_temp

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
        height=360,
        margin=dict(l=50, r=50, t=90, b=20),
    )
    return fig


def generate_tab_graph(item, patients, **kwargs):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    symbols = ['circle', 'x', 'star-triangle-up', 'star-triangle-down', 'star']
    colours = ['rgb(100, 143, 255)', 'rgb(120, 94, 240)', 'rgb(220, 38, 127)', 'rgb(254, 97, 0)',
               'rgb(255, 176, 0)']

    table_item = df_events.query(f'itemid == {item}')
    if table_item.empty:
        return {}
    if not pd.to_numeric(table_item['value'], errors='coerce').isnull().all():
        table_item['value'] = pd.to_numeric(table_item['value'], errors='coerce')

    units = list(table_item['valueuom'])[0]
    for i, each_patient in enumerate(patients):
        table_item_patient = table_item.query(f'subject_id == {each_patient}')
        if table_item_patient.empty:
            continue
        start_date = min(table_item_patient['charttime'])
        end_date = start_date + timedelta(hours=96)

        mask = (table_item_patient['charttime'] > start_date) & (table_item_patient['charttime'] <= end_date)
        table_item_patient = table_item_patient.loc[mask]
        table_item_patient = table_item_patient.sort_values(by="charttime")

        def charttime_to_deltatime(charttime_series):
            deltatime_series = charttime_series.apply(lambda x: abs(x - start_date).total_seconds() / 3600)
            return deltatime_series

        table_item_patient['charttime'] = charttime_to_deltatime(table_item_patient['charttime'])

        if table_item.isna().sum().sum() / table_item.shape[0] > 0.5:  # text data
            if i == 0:
                fig.add_trace(
                    go.Scatter(x=table_item_patient["charttime"], y=table_item_patient["value"], mode='markers',
                               name=f"Patient {each_patient}", hovertemplate='%{y}<extra></extra>',
                               marker=dict(color=colours[i])),
                    secondary_y=False,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=table_item_patient["charttime"], y=table_item_patient["value"], mode='markers',
                               name=f"Patient {each_patient}", hovertemplate='%{y}<extra></extra>',
                               marker=dict(color=colours[i]),
                               visible='legendonly'),
                    secondary_y=False,
                )
        else:  # numerical data
            if i < 3:
                fig.add_trace(
                    go.Scatter(x=table_item_patient["charttime"], y=table_item_patient["value"], mode='lines+markers',
                               name=f"Patient {each_patient}", hovertemplate='%{y:.1f}<extra></extra>',
                               marker=dict(color=colours[i])),
                    secondary_y=False,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=table_item_patient["charttime"], y=table_item_patient["value"], mode='lines+markers',
                               name=f"Patient {each_patient}", hovertemplate='%{y:.1f}<extra></extra>',
                               marker=dict(color=colours[i]),
                               visible='legendonly'),
                    secondary_y=False,
                )

        fig['data'][i]['marker']['symbol'] = symbols[i]

    del table_item_patient
    del mask

    if kwargs['config'].spikes:
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                         spikemode='across+marker')
        fig.update_yaxes(showspikes=True,
                         spikecolor="black", spikesnap="cursor", spikethickness=1, spikedash='dot',
                         spikemode='across+marker', secondary_y=False)

    fig.update_yaxes(title_text=f"Value ({units})")
    fig.update_layout(
        xaxis_title="Time (Hour)",
        font=dict(
            family=kwargs['config'].text_font,
            size=kwargs['config'].text_size,
            color=kwargs['config'].text_color
        ),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.47, bgcolor='rgba(0,0,0,0)'),
        legend_tracegroupgap=1,
        margin=dict(l=50, r=0, t=40, b=40),
        height=360,
    )
    return fig


def query_patients(item):
    target_table = df_events.query(f'itemid == {item}')
    if target_table.empty:
        return []

    patients_sorted = Counter(list(target_table['subject_id'])).most_common()
    ranked_patients = [i[0] for i in patients_sorted]

    return ranked_patients


def initialize_download_button(annotated):
    if annotated:
        return False
    return True


def initialize_all_patients_graph():
    items = [{"label": f'{each_id}: {itemsid_dict[each_id]}', "value": each_id} for each_id in unannotated_list]
    fig = generate_all_patients_graph(items[0]["value"], config=config.kwargs)
    return fig


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


def annotate(item, annotation, ontology, comments, skipped=False):
    if skipped:
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     'ontology': ontology,
                     'annotatedid': 'Skipped',
                     'annotatedlabel': 'n/a',
                     'comments': comments
                     }
    else:
        # item_row = df_items.query(f'itemid == {item}')
        item_dict = {'itemid': item,
                     'label': itemsid_dict[item],
                     'ontology': ontology,
                     # 'mimic_loinc': item_row['loinc_code'].item(),      # not available in mimic-iv-v2.0, removed in d-items
                     'annotatedid': [each_id['CODE'] for each_id in annotation],
                     'annotatedlabel': [each_id['LABEL'] for each_id in annotation],
                     'comments': comments
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


def filter_datatable(data, filter_search, scorer, ontology_filter):
    if not filter_search:
        new_data = data
    else:
        if ontology_filter == 'loinc':
            new_data = [row for row in data if row['CLASS'] in filter_search]
        elif ontology_filter == 'snomed':
            new_data = [row for row in data if row['HIERARCHY'] in filter_search]
        else:
            raise InvalidOntology
    df_newdata = pd.DataFrame.from_records(new_data)
    tooltip_dict = df_newdata[scorer].to_dict()
    tooltip_outputs = []
    for each_value in tooltip_dict:
        tooltip_output = {
            'value': f'**{tooltip_dict[each_value]}**',
            'type': 'markdown'}
        tooltip_outputs.append({'RELEVANCE': tooltip_output})
    return new_data, tooltip_outputs


def get_n_values(item, n):
    table = df_events.query(f'itemid == {item}')
    if table.empty:
        n_values = ['None']
    else:
        table['value_with_uom'] = table['value'] + table['valueuom']
        top_in_table = table['value_with_uom'].value_counts().nlargest(n)
        n_values = top_in_table.index.values
    return n_values


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
        Input("skip-checklist", "value"),
    ]
)
def enable_submit_button(ontology, skip):
    if ontology or 'skip' in skip:
        return False
    return True


@app.callback(
    Output("download-outer", "hidden"),
    [
        Input("submit-btn", "n_clicks"),
    ],
)
def enable_download_button(_):
    if annotated_list:
        return False
    return True


@app.callback(
    Output("comments-text", "value"),
    Output("skip-checklist", "value"),
    [
        Input("submit-btn", "n_clicks"),
        Input("skip-checklist", "value"),
    ],
    prevent_initial_call=True,
)
def clear_reset_skip(_, __):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return '', []
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
            for folderName, subfolders, filenames in os.walk(PATH_results):
                for filename in filenames:
                    if '.git' in filename:
                        continue
                    zipObj.write(os.path.join(folderName, filename))

    return dcc.send_bytes(write_archive, "annotations.zip")


@app.callback(
    Output("related-datatable", "active_cell"),
    Output("related-datatable", "selected_cells"),
    [
        Input("submit-btn", "n_clicks"),
        Input("ontology-select", "value"),
        Input("ontology-datatable", "data_previous"),
    ],
    [
        State("ontology-datatable", "data"),
        State("item-select", "value"),
        State("skip-checklist", "value"),
        State("comments-text", "value"),
    ],
    prevent_initial_call=True
)
def reset_annotation(_, ontology, prev_ontology_data, curr_ontology_data, item,
                     skip, comments):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        if 'skip' in skip:
            annotate(item, None, ontology, comments, skipped=True)
        else:
            annotate(item, curr_ontology_data, ontology, comments)
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
    if ontology is None:
        raise PreventUpdate
    scorer_options = generate_scorer_options(ontology)
    return scorer_options[0]["value"], scorer_options


@app.callback(
    Output("item-select", "options"),
    Output("item-select", "value"),
    [
        Input("submit-btn", "n_clicks"),
    ],
    [
        State("item-select", "options"),
        State("item-select", "value"),
        State("skip-checklist", "value"),
    ]
)
def update_item_dropdown(_, options, value, skip):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == '.':
        raise PreventUpdate
    new_value = value
    if triggered_id == 'submit-btn.n_clicks':
        if 'skip' in skip:
            if value in unannotated_list:
                unannotated_list.remove(value)
            skipped_list.append(value)
            options, new_value = update_item_options(options, value, skipped=True)
        elif value in unannotated_list:
            unannotated_list.remove(value)
            options, new_value = update_item_options(options, value)
    return options, new_value


@app.callback(
    Output("tabs", "value"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
    ]
)
def update_tabs_view(item, _):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return "home-tab"
    # elif item is not None:
    #     raise PreventUpdate
    else:
        return "home-tab"


@app.callback(
    Output("all_patients_graph", "figure"),
    Output("patient_tab_graph", "figure"),
    Output("patient_tab", "disabled"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_graph(item, _):
    disabled = True

    if item is None:
        return {}, {}, disabled

    patients = query_patients(item)
    if patients:
        disabled = False
        if len(patients) > 5:
            top_patients = patients[0:5]
        else:
            top_patients = patients
        patient_tab = (generate_tab_graph(item, top_patients, config=config.kwargs), disabled)
    else:
        patient_tab = ({}, True)
    all_patients_graph = generate_all_patients_graph(item, config=config.kwargs)

    return all_patients_graph, patient_tab[0], patient_tab[1]


@app.callback(
    Output("related-datatable", "page_current"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("search-btn", "n_clicks"),
    ]
)
def reset_related_datatable_page(item, _, __):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return 0
    elif triggered_id == 'search-btn.n_clicks':
        return 0
    if not item:
        return 0


@app.callback(
    Output("ontology-datatable", "data"),
    Output("ontology-datatable", "columns"),
    Output("ontology-datatable", "tooltip_data"),
    [
        Input("submit-btn", "n_clicks"),
        Input("related-datatable", "active_cell"),
    ],
    [
        State("related-datatable", "data"),
        State("ontology-datatable", "data"),
        State("ontology-datatable", "columns"),
        State("ontology-select", "value"),
        State("skip-checklist", "value"),
    ],
    prevent_initial_call=True,
)
def update_ontology_datatable(_, related, curr_data_related, curr_data_ontology, curr_ontology_cols, ontology, skip):
    triggered_ids = dash.callback_context.triggered
    if triggered_ids[0]['prop_id'] == '.':
        raise PreventUpdate
    if 'skip' in skip:
        raise PreventUpdate
    if triggered_ids[0]['prop_id'] == 'submit-btn.n_clicks':
        if curr_data_ontology is None:
            return None, curr_ontology_cols, []
        return curr_data_ontology[0:0], curr_ontology_cols, []

    df_ontology = sql_searchers[ontology].get_all_ontology_no_data()
    if not curr_data_ontology:
        df_data = pd.DataFrame(columns=df_ontology.columns)
    else:
        df_data = pd.DataFrame.from_records(curr_data_ontology)
    columns = [{"name": 'CODE', "id": 'CODE'}, {"name": 'LABEL', "id": 'LABEL'}]

    if related:
        temp_dict = [d for d in curr_data_related if d.get('id') == related['row_id']][0]
        if curr_data_ontology:
            if temp_dict['CODE'] in [each_selected['CODE'] for each_selected in curr_data_ontology]:
                raise PreventUpdate
        df_data = pd.concat(
            [df_data, df_ontology.loc[df_ontology['CODE'] == temp_dict['CODE']]])

    def table_gen(each_row):
        try:
            df_tooltip = pd.DataFrame()
            for each_ontology in list_of_ontologies:
                df_tooltip = sql_searchers[ontology].search_ontology_by_code(f'\"{each_row["CODE"]}\"')
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
    del tooltip_tables

    tooltip_outputs = []
    for each_dict in tooltip_dicts:
        tooltip_output = {
            'value': "\n\n".join([f'**{each_item}**: {each_dict[each_item]}' for each_item in each_dict.keys()
                                  if each_item not in ['CODE', 'COMPONENT', 'LABEL']]),
            'type': 'markdown'}
        tooltip_outputs.append({'LABEL': tooltip_output})
    data = df_data.to_dict('records')
    return data, columns, tooltip_outputs


@app.callback(
    Output("filter-search", "options"),
    Output("filter-search", "placeholder"),
    [
        Input("store-search-results", "data"),
    ],
    [
        State("ontology-select", "value"),
    ]
)
def update_filter_search_dropdown(data, ontology):
    if not data:
        return [], "No Classes Available"
    if ontology == 'loinc':
        filters = set([i['CLASS'] for i in data])
        placeholder = "Select Class..."
    elif ontology == 'snomed':
        filters = set([i['HIERARCHY'] for i in data])
        placeholder = "Select Hierarchy..."
    else:
        raise InvalidOntology
    options = [{"label": each_filter, "value": each_filter} for each_filter in filters]
    return options, placeholder


@app.callback(
    Output("related-datatable", "data"),
    Output("related-datatable", "columns"),
    Output("related-datatable", "tooltip_data"),
    Output("search-input", "value"),
    Output("store-search-results", "data"),
    [
        Input("item-select", "value"),
        Input("submit-btn", "n_clicks"),
        Input("scorer-select", "value"),
        Input("ontology-select", "value"),
        Input("search-btn", "n_clicks"),
        Input("filter-search", "value"),
    ],
    [
        State("search-input", "value"),
        State("store-search-results", "data"),
        State("list-suggested-inputs", "children"),
    ]
)
def update_related_datatable(item, _, scorer, ontology_filter, __, filter_search, search_string, init_data,
                             suggestions):
    if not item:
        return None, [{'name': 'Invalid Source Item', 'id': 'invalid'}], [], '', None

    listed_options = [option['props']['value'] for option in suggestions]

    df_ontology = sql_searchers[ontology_filter].get_all_ontology_no_data()

    triggered_id = dash.callback_context.triggered[0]['prop_id']

    # if triggered_id == 'search-input.value':
    #     if search_string not in listed_options:
    #         raise PreventUpdate

    if triggered_id == 'filter-search.value':
        filtered_data, tooltip_output = filter_datatable(init_data, filter_search, scorer, ontology_filter)
        return filtered_data, no_update, tooltip_output, no_update, no_update

    if ontology_filter is None:
        raise PreventUpdate

    query = itemsid_dict[item]
    df_data = search_ontology(
        query=query,
        df_ontology=df_ontology,
        method=scorer,
        n=50,
        indexes=my_indexes,
        sql_searcher=sql_searchers[ontology_filter],
        triggered_id=triggered_id,
        listed_options=listed_options,
        ontology_filter=ontology_filter,
        search_string=search_string
    )

    if isinstance(df_data, str):
        return_string = df_data
        return None, [{'name': 'No Results Found', 'id': 'none'}], [], return_string, None

    if len(df_data.index) > 50:
        df_data = df_data.iloc[:50]

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

    if config.ontology.rank:
        metadata = {'examples': get_n_values(item, n=3)}  # target concept metadata, can add other info
        data = rank(target=itemsid_dict[item], choices=data, method=config.ontology.rank, metadata=metadata)

    return data, return_columns, tooltip_outputs, query, data


@app.callback(
    Output("hidden-div", "children"),
    Output("refresh-url", "href"),
    # Output("df_items-store", "data"),
    # Output("df_events-store", "data"),
    # Output("df_ontology-store", "data"),
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
    global list_of_ontologies
    global df_items, df_events
    global itemsid_dict
    global PATH_data, PATH_items, PATH_results, PATH_ontology
    global annotated_list, skipped_list, unannotated_list
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'yaml' in filename:
            config = yaml.unsafe_load(decoded)
        else:
            raise ConfigurationFileError
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


@app.callback(
    Output("search-btn", "style"),
    Output("search-btn", "disabled"),
    Output("store-search-query", "data"),
    [
        Input("submit-btn", "n_clicks"),
        Input("search-btn", "n_clicks"),
        Input("search-input", "value"),
    ],
    [
        State("store-search-query", "data"),
    ],
    prevent_initial_call=True,
)
def change_search_button_style(_, __, new_query, curr_query):
    style = {'width': '100%', 'color': 'grey', 'margin-top': '15px'}
    if curr_query is None:
        return no_update, True, new_query
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks':
        return style, True, new_query
    elif triggered_id == 'search-btn.n_clicks':
        return style, True, new_query
    else:
        if new_query != curr_query:
            style = {'width': '100%', 'color': 'white', 'margin-top': '15px'}
        return style, False, no_update


@app.callback(
    Output("list-suggested-inputs", "children"),
    [
        Input("item-select", "value"),
        Input("ontology-select", "value"),
    ],
)
def generate_suggestions(item, ontology):
    query = itemsid_dict[item]
    df_data = my_indexes[ontology].get_search_results(query)
    if df_data.empty:
        return []

    options = [html.Option(value=label) for label in df_data['LABEL'].tolist()]
    return options
