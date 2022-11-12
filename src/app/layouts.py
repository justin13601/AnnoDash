from src.app.callbacks import *


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
                                title="Getting Started",
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
                            'margin-top': '-2px'
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
                id="skip-outer",
                hidden=False,
                style={'margin-top': '11px'},
                children=[
                    dcc.Checklist(
                        id='skip-checklist',
                        options=[
                            {'label': 'Skip', 'value': 'skip'},
                        ],
                        value=[],
                        style={'width': '16%', 'color': 'white', 'textAlign': 'left',
                               'verticalAlign': 'center', 'float': 'left', 'margin-top': '6px'},
                    ),
                    html.Div(id='comments-outer',
                             hidden=False,
                             children=[
                                 dcc.Input(
                                     id="comments-text",
                                     placeholder="Comments",
                                     debounce=True,
                                     style={"width": '83%', 'margin-left': '0px', 'float': 'right'},
                                     autoFocus=True,
                                     disabled=False
                                 ),
                             ]),
                ],
            ),
            html.Div(
                id="submit-btn-outer",
                hidden=False,
                children=[
                    dcc.Loading(
                        id="submit-loading",
                        type="dot",
                        color='#2c89f2',
                        children=[
                            html.Button(id="submit-btn", children="Submit & Next", n_clicks=0,
                                        style={'width': '100%', 'color': 'white',
                                               'margin-top': '11px', 'margin-bottom': '6px'},
                                        disabled=False),
                        ],
                        style={'margin-top': '45px'}
                    ),
                ],
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
            dcc.Store(id='store-search-results'),
            dcc.Store(id='store-search-query'),
            # dcc.Store(id='df_ontology-store'),
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
            html.Div(
                hidden=True,
                children=[
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
                style={
                    'padding-top': '10px'
                },
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
                                        dcc.Tab(label='Sample Records\n(Individual Patients)', id="patient_tab",
                                                disabled=False,
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
                                                        id="patient_tab_graph",
                                                        figure={}
                                                    )
                                                ]),
                                    ], id='tabs', value='home-tab', style={'height': '75px'}),
                                ],
                            ),
                        ],
                        style={
                            'background-color': 'rgba(0, 0, 0, 0)'
                        },
                    ),
                ]
            ),
            html.Div(
                className='results-card',
                children=[
                    html.Hr(
                        style={
                            'margin-top': '10px',
                            'margin-bottom': '7px'
                        }
                    ),
                ],
            ),
            html.Div(
                id='results-card',
                className='results-card',
                children=[
                    # search column
                    html.Div(
                        id="search-column",
                        children=[
                            html.Div(
                                id='search-ontology-outer',
                                children=[
                                    html.P(
                                        style={'margin-top': '1px'},
                                        children=[
                                            html.B('Search Parameters:'),
                                        ]),
                                    html.Div(id='search-outer',
                                             hidden=False,
                                             children=[
                                                 html.Div(
                                                     className="datalist",
                                                     children=[
                                                         html.Datalist(
                                                             id='list-suggested-inputs',
                                                             children=generate_suggestions(unannotated_list[0],
                                                                                           list_of_ontologies[0])
                                                         ),
                                                     ],
                                                 ),
                                                 dcc.Input(
                                                     id="search-input",
                                                     type='text',
                                                     list='list-suggested-inputs',
                                                     placeholder="",
                                                     debounce=False,
                                                     style={"width": '100%', 'margin-left': '0px'},
                                                     autoFocus=True,
                                                     disabled=False
                                                 ),
                                             ]),
                                    html.Div(style={'margin-top': '15px'}),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='filter-search',
                                            multi=True,
                                            placeholder="Select Class...",
                                            style={'border-radius': 0, 'color': 'black'},
                                        ),
                                    ],
                                        style={'height': '135px', 'overflow-y': 'auto', 'background-color': 'white'}),
                                    dcc.Loading(
                                        id="related-loading",
                                        type="dot",
                                        color='#2c89f2',
                                        children=[
                                            html.Button(id="search-btn", children="Search", n_clicks=0,
                                                        style={'width': '100%', 'color': 'grey', 'margin-top': '15px'},
                                                        disabled=False),
                                        ],
                                        style={'margin-top': '20px'}
                                    ),
                                ],
                            ),
                        ],
                        style={
                            'float': 'left',
                            'width': '20%',
                        },
                    ),
                    # results column
                    html.Div(
                        id="search-results-column",
                        style={
                            'margin-left': '2rem',
                            'width': '75%',
                            'float': 'right',
                        },
                        children=[
                            html.Div(
                                id="annotation-outer",
                                hidden=False,
                                children=[
                                    html.Div(
                                        className='loading-wrapper',
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
                                                style={'margin-top': '1px'},
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
                                                                              fixed_rows={'headers': True},
                                                                              filter_action='native',
                                                                              filter_options={
                                                                                  'case': 'insensitive'},
                                                                              style_data={
                                                                                  'width': 'auto',
                                                                                  'maxWidth': '100px',
                                                                                  'minWidth': '100px',
                                                                                  'whiteSpace': 'normal'
                                                                              },
                                                                              style_table={
                                                                                  'height': '240px',
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
                                                                                      'if': {
                                                                                          'column_id': 'RELEVANCE'},
                                                                                      'width': '1%',
                                                                                      'maxWidth': '1%',
                                                                                      'minWidth': '1%',
                                                                                  },
                                                                                  {
                                                                                      'if': {
                                                                                          'column_id': 'CODE'},
                                                                                      'width': '10%',
                                                                                      'maxWidth': '10%',
                                                                                      'minWidth': '10%',
                                                                                  }
                                                                              ],
                                                                              # page_size=20,
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
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="download-outer",
                        hidden=initialize_download_button(annotated_list),
                        children=[
                            html.Button(id="download-btn", children="Download current annotations.zip", n_clicks=0,
                                        style={'width': '100%', 'color': 'white', 'margin-top': '10px'},
                                        disabled=False),
                            dcc.Download(id="download-annotations"),
                        ],
                    ),
                ]
            ),
        ],
    )

