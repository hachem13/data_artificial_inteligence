from dash.dependencies import Input, Output
import plotly.express as px
from app import app


@app.callback(
    Output('id1', 'children'),
    Input('plot1', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('id2', 'children'),
    Input('image', 'value'))
def display_value1(value):
    return 'You have selected "{}"'.format(value)


