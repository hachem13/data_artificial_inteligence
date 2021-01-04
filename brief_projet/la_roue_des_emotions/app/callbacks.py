from dash.dependencies import Input, Output
import plotly.express as px
from app import app
import dash
from layouts import layout1, layout2, layout3, layout4, pipe

import pandas as pd

@app.callback(
    Output('id1', 'children'),
    Input('Home page', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('id2', 'children'),
    Input('kaggle', 'value'))
def display_value1(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('id3', 'children'),
    Input('Kaggle1', 'value'))
def display_value2(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output("output", "children"),
    Input("input1", "value"),)
def update_output(input1):  
    
    text = [input1]
    if input1 is None:
        return ""
    else:
        y_pred = pipe.predict(text)
        return u'Emotion : {}'.format(y_pred)
    