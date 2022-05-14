#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import csv
from google.cloud import bigquery
import numpy as np
import pandas as pd
import matplotlib
import plotly
from dash import Dash, html


def load_data(path):
    filename = os.path.basename(path).strip()
    print('Loading ' + filename + '...')
    df_data = pd.read_csv(path)
    print('Done.')
    return df_data


def big_query(query):
    client = bigquery.Client()
    query_job = client.query(query)  # API request
    print("The query data:")
    for row in query_job:
        # row values can be accessed by field name or index
        print("name={}, count={}".format(row[0], row["total_people"]))
    return


# run
if __name__ == '__main__':
    df_labitems = load_data('/home/justinxu/PycharmProjects/mimic-iv-dash/D_LABITEMS.csv')
    df_labevents = load_data('/home/justinxu/PycharmProjects/mimic-iv-dash/LABEVENTS.csv')
