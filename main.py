#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""
import dash
from dash import Dash, html, dcc, dash_table, ALL, ctx
from dash.dependencies import State, Input, Output, ClientsideFunction

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

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.Div(
            id='dashboard-content',
            children=[],
        )
    ]
)
# run app.py (MIMIC-Dash v2)
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
    # app.run_server(port=8888, debug=True, use_reloader=False)
