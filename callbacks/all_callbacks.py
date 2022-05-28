import dash
from callbacks_manager import CallbackManager
from dash.dependencies import State, Input, Output, ClientsideFunction

callback_manager = CallbackManager()


@callback_manager.callback(
    Output("wait_time_table", "children"),
    [
        Input("date-picker-select", "start_date"),
        Input("date-picker-select", "end_date"),
        Input("labitem-select", "value"),
        Input("admit-select", "value"),
        Input("patient_volume_hm", "clickData"),
        Input("reset-btn", "n_clicks"),
    ]
    + wait_time_inputs
    + score_inputs,
)
def update_table(start, end, clinic, admit_type, heatmap_click, reset_click, *args):
    start = start + " 00:00:00"
    end = end + " 00:00:00"

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    triggered_value = None
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
        triggered_value = ctx.triggered[0]["value"]

    # filter data
    filtered_df = df[
        (df["Clinic Name"] == clinic) & (df["Admit Source"].isin(admit_type))
        ]
    filtered_df = filtered_df.sort_values("Check-In Time").set_index("Check-In Time")[
                  start:end
                  ]
    departments = filtered_df["Department"].unique()

    # Highlight click data's patients in this table
    if heatmap_click is not None and prop_id != "reset-btn":
        hour_of_day = heatmap_click["points"][0]["x"]
        weekday = heatmap_click["points"][0]["y"]
        clicked_df = filtered_df[
            (filtered_df["Days of Wk"] == weekday)
            & (filtered_df["Check-In Hour"] == hour_of_day)
            ]  # slice based on clicked weekday and hour
        departments = clicked_df["Department"].unique()
        filtered_df = clicked_df

    # range_x for all plots
    wait_time_xrange = [
        filtered_df["Wait Time Min"].min() - 2,
        filtered_df["Wait Time Min"].max() + 2,
    ]
    score_xrange = [
        filtered_df["Care Score"].min() - 0.5,
        filtered_df["Care Score"].max() + 0.5,
    ]

    figure_list = []

    if prop_type != "selectedData" or (
            prop_type == "selectedData" and triggered_value is None
    ):  # Default condition, all ""

        for department in departments:
            department_wait_time_figure = create_table_figure(
                department, filtered_df, "Wait Time Min", wait_time_xrange, ""
            )
            figure_list.append(department_wait_time_figure)

        for department in departments:
            department_score_figure = create_table_figure(
                department, filtered_df, "Care Score", score_xrange, ""
            )
            figure_list.append(department_score_figure)

    elif prop_type == "selectedData":
        selected_patient = ctx.triggered[0]["value"]["points"][0]["customdata"]
        selected_index = [ctx.triggered[0]["value"]["points"][0]["pointIndex"]]

        # [] turn on un-selection for all other plots, [index] for this department
        for department in departments:
            wait_selected_index = []
            if prop_id.split("_")[0] == department:
                wait_selected_index = selected_index

            department_wait_time_figure = create_table_figure(
                department,
                filtered_df,
                "Wait Time Min",
                wait_time_xrange,
                wait_selected_index,
            )
            figure_list.append(department_wait_time_figure)

        for department in departments:
            score_selected_index = []
            if department == prop_id.split("_")[0]:
                score_selected_index = selected_index

            department_score_figure = create_table_figure(
                department,
                filtered_df,
                "Care Score",
                score_xrange,
                score_selected_index,
            )
            figure_list.append(department_score_figure)

    # Put figures in table
    table = generate_patient_table(
        figure_list, departments, wait_time_xrange, score_xrange
    )
    return table
