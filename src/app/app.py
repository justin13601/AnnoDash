#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 10, 2022
@author: Justin Xu
"""
import dash

######################################################################################################
# APP #
######################################################################################################

assets_path = '../../assets'

app = dash.Dash(
    __name__,
    assets_folder=assets_path,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)

app.title = "MIMIC-Dash: A Clinical Terminology Annotation Dashboard"
app.config.suppress_callback_exceptions = True
