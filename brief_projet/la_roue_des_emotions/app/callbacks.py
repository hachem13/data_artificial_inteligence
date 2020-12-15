from dash.dependencies import Input, Output
import plotly.express as px
from app import app
import dash
from layouts import layout1, layout2

import pandas as pd



@app.callback(
    Output('id1', 'children'),
    Input('kaggle', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('id2', 'children'),
    Input('Kaggle1', 'value'))
def display_value1(value):
    return 'You have selected "{}"'.format(value)
