import dash
from callbacks.callback_manager import CallbackManager
from dash.dependencies import State, Input, Output, ClientsideFunction
from app import *

callback_manager = CallbackManager()


@callback_manager.callback(
    Output('submit-btn-outer', 'hidden'),
    [
        Input('annotate-select', 'value'),
    ]
)
def enable_submit_button(annotation):
    if annotation:
        return False
    return True


@callback_manager.callback(
    Output("submit-btn", "n_clicks"),
    Output('annotate-select', 'value'),
    # Output('annotate-text', 'value'),
    Output('confirm-replace', 'displayed'),
    [
        Input("submit-btn", "n_clicks"),
    ],
    [
        State('labitem-select', 'value'),
        State('annotate-select', 'value'),
        # State('annotate-text', 'value')
    ]
)
def submit_annotation(n_clicks, labitem, annotation):
    if n_clicks == 0:
        raise PreventUpdate
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if n_clicks > 0 and triggered_id == 'submit-btn.n_clicks':
        annotate(labitem, annotation)
        n_clicks = 0
        return n_clicks, '', False


@callback_manager.callback(
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


@callback_manager.callback(
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


@callback_manager.callback(
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


@callback_manager.callback(
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
        return generate_all_patients_graph(labitem, config=config['graphs']['kwargs']), \
               generate_tab_graph(labitem, patient, bg_pair), disabled, \
               generate_tab_graph(labitem, patient, chem_pair), disabled, \
               generate_tab_graph(labitem, patient, cbc_pair), disabled

    return generate_all_patients_graph(labitem,
                                       config=config['graphs']['kwargs']), {}, disabled, {}, disabled, {}, disabled


@callback_manager.callback(
    Output("search-datatable-outer", "hidden"),
    Output("loinc-datatable", "data"),
    Output("loinc-datatable", "columns"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_datatable(annotation, submit):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks' or annotation is None:
        return True, None, []
    df_data = df_loinc.loc[df_loinc['LOINC_NUM'] == annotation]
    data = df_data.to_dict('records')
    columns = [{"name": i, "id": i} for i in df_data.columns]
    return False, data, columns


@callback_manager.callback(
    Output("related-datatable-outer", "hidden"),
    Output("related-loinc-datatable", "data"),
    [
        Input("annotate-select", "value"),
        Input("submit-btn", "n_clicks"),
    ],
)
def update_related_datatable(annotation, submit):
    triggered_id = dash.callback_context.triggered[0]['prop_id']
    if triggered_id == 'submit-btn.n_clicks' or annotation is None:
        return True, None

    data = df_loinc_new[0:1000].to_dict('records')
    return False, data
