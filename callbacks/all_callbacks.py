import dash
from callbacks.callback_manager import CallbackManager
from dash.dependencies import State, Input, Output, ClientsideFunction
from app import *

callback_manager = CallbackManager()


@callback_manager.callback(
    Output('submit-btn', 'disabled'),
    [
        Input('item-select', 'value'),
    ]
)
def enable_button(item):
    pass
