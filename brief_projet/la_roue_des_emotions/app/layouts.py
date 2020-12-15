import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import numpy as np
from app import app
import plotly.graph_objs as go
import plotly.express as px
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from  sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import seaborn as sns
from time import time
from collections import defaultdict
from pipline import print_table
from joblib import load



df = pd.read_csv('data/Emotion_final.csv')
df1 = pd.read_csv('data/text_emotion.csv')


# Grouped column Emotion
grouped = df.groupby("Emotion").describe()
# ascending order for emotions 
grouped = grouped.sort_values(by=[('Text','count')] , ascending = False)
    
grouped1 = df1.groupby("sentiment").describe()
# ascending order for emotions 
grouped1 = grouped1.sort_values(by=[('tweet_id','count')] , ascending = False)

# Process df1
corpus = df.Text
targets = df.Emotion



vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

# Compute rank
words = vectorizer.get_feature_names()
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

def subsample(x):
    return np.hstack((x[:30]))


freq = subsample(wrank)
r = np.arange(len(freq))

# Process df2
corpus1 = df1.content
targets1 = df1.sentiment

vectorizer1 = CountVectorizer()

X1 = vectorizer1.fit_transform(corpus1)

words1 = vectorizer1.get_feature_names()
wsum1 = np.array(X1.sum(0))[0]
ix1 = wsum1.argsort()[::-1]
wrank1 = wsum1[ix1] 
labels1 = [words1[i] for i in ix1]

freq1 = subsample(wrank1)
r1 = np.arange(len(freq1))


colors = {
    'background': '#00000',
    'text': '#111111'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



html.Br(),

def Header():
    return html.Div([
        get_header(),
        html.Br([]),
        get_menu()
    ])

def get_header():
    header = html.Div([

        html.Div([
            html.H1(
                'Roue des Emotions')
        ], className="twelve columns padded")

    ], className="row")
    return header

def get_menu():
    menu = html.Div([

        dcc.Link('page 1        |', href='/apps/page1', className="tab first"),

        dcc.Link('page 2        |', href='/apps/page2', className="tab")

    ], className="row")
    return menu

# plot

fig = go.Figure(data=[
    go.Bar(name='frequence des emotions',
            x=grouped.index, 
            y= grouped[('Text', 'count')], 
            marker = dict(color = 'rgba(153, 181, 71, 0.5)',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = grouped[('Text', 'count')])
    ])
fig.update_layout(barmode='group',title = 'Fréquence des emotions ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))

fig1 = go.Figure(data=[
    go.Bar(name='frequence des emotions',
            x=grouped.index, 
            y= grouped1[('tweet_id', 'count')], 
            marker = dict(color = 'rgba(174, 204, 161, 0.5)',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = grouped1[('tweet_id', 'count')])
    ])
fig1.update_layout(barmode='group',title = 'Fréquence des emotions ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))

fig2 = go.Figure(data=[
    go.Bar(name= 'frequence des mots',
    x = r,
    y = freq, 
    marker = dict(color = 'rgba(153, 181, 71, 0.5)',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = subsample(labels))
])
fig2.update_layout(barmode='group',title = 'Fréquence des mots ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))
fig2.update_xaxes(
        tickmode='array',
        tickvals = r,
        ticktext = labels
)

fig3 = go.Figure(data=[
    go.Bar(name= 'frequence des mots',
    x = r1,
    y = freq1, 
    marker = dict(color = 'rgba(153, 181, 71, 0.5)',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = subsample(labels1))
])
fig3.update_layout(barmode='group',title = 'Fréquence des mots ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))
fig3.update_xaxes(
        tickmode='array',
        tickvals = r1,
        ticktext = labels1
)



layout1 =  html.Div([
    Header(),
    html.Br(),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label='Kaggle Data', children=[
            dash_table.DataTable(
            id='kaggle',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            editable=False,
            css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}],
            style_table={'overflowX': 'scroll',
                         'overflowY': 'scroll',
                         'maxHeight': '300px',
                         'maxWidth': '1500px'},
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'},
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'center'
                } for c in ['Date', 'Region']
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#a1b5cc',
                    'color': 'white'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color' : 'white',
                'fontWeight': 'bold'
                },
            ),
            dcc.Graph(
                id = 'plot',
                figure= fig
            ),
            dcc.Graph(
                id = 'plot2',
                figure= fig2
            )]),
        dcc.Tab(label='Data World', children=[
            dash_table.DataTable(
            id='Data_world',
            columns=[{"name": i, "id": i} for i in df1.columns],
            data=df1.to_dict('records'),
            editable=False,
            css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}],
            style_table={'overflowX': 'scroll',
                         'overflowY': 'scroll',
                         'maxHeight': '300px',
                         'maxWidth': '1500px'},
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'},
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'center'
                } for c in ['Date', 'Region']
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#b2cca1',
                    'color': 'white'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color' : 'white',
                'fontWeight': 'bold'
                },
            ),
            dcc.Graph(
                figure=fig1
                ),
            dcc.Graph(
                figure=fig3
                )
            ])
        
    ]),
    html.Br(),
    html.Br(),
    html.Div(id='id1'),
    dcc.Link('Go to page2', href='/apps/page2')
])

printTable = print_table(load('filename.joblib'))


layout2 = html.Div([
    Header(),
    html.Div([
    dcc.Tab(label='Kaggle Data', children=[
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable.columns],
            data=printTable.to_dict('records'),
            editable=False,
            )
        ]),    
    ]), 
    html.Br(),
    html.Br(),
    html.Div(id='id2'),
    dcc.Link('Go to page1', href='/apps/page1')
])            