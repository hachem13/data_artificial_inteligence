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
from time import time
from collections import defaultdict
import pickle








df = pd.read_csv('data/Emotion_final.csv')
df1 = pd.read_csv('data/text_emotion.csv')
# concate data set 
df2 = df.append(df1[['content','sentiment']].rename(columns={"content": "Text", "sentiment": "Emotion"}),ignore_index=True)
df2.Emotion = df2.Emotion.replace(['happiness', 'worry'], ['happy', 'fear'])


# Grouped column Emotion
grouped = df.groupby("Emotion").describe()
# ascending order for emotions 
grouped = grouped.sort_values(by=[('Text','count')] , ascending = False)
    
grouped1 = df1.groupby("sentiment").describe()
# ascending order for emotions 
grouped1 = grouped1.sort_values(by=[('tweet_id','count')] , ascending = False)

grouped2 = df2.groupby("Emotion").describe()
# ascending order for emotions 
grouped2 = grouped2.sort_values(by=[('Text','count')] , ascending = False)


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

# Process df1
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

# Process df1
corpus2 = df2.Text
targets2 = df2.Emotion

vectorizer2 = CountVectorizer()

X2 = vectorizer2.fit_transform(corpus2)

words2 = vectorizer2.get_feature_names()
wsum2 = np.array(X2.sum(0))[0]
ix2 = wsum2.argsort()[::-1]
wrank2 = wsum2[ix2] 
labels2 = [words2[i] for i in ix2]

freq2 = subsample(wrank2)
r2 = np.arange(len(freq2))


colors = {
    'background': '#00000',
    'text': '#111111'
}

external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"]

# confusion_matrix1 = "assets/confusion-matrix.png" # replace with your own image
# encoded_image1 = base64.b64encode(open(confusion_matrix1, 'rb').read())
# html.Img(src='data:image/png;base64,{}'.format(encoded_image1))

html.Br(),

def Header():
    return html.Div([
        html.Br([]),
#        get_menu(),
        html.Br([]),
        html.Br([]),

    ])
 

# def get_menu():
#     navbar = dbc.NavbarSimple(
#         children=[
#             html.Div([html.Img(src='assets/emotions.png', height = "100px")]),
#             dbc.NavItem(dbc.NavLink("Home", href="/apps/page1")),
#             dbc.DropdownMenu(
#                 children=[
#                     dbc.DropdownMenuItem("More pages", header=True),
#                     dbc.DropdownMenuItem("Résultat Logs", href="/apps/page2"),
#                 ],
#                 nav=True,
#                 in_navbar=True,
#                 label="More",
#             ),
#         ],
#         brand="La Roue des Emotions",className="H1", 
#         color="primary",
#         dark=True,)
#     return navbar

#def get_menu():
    # dropdown = dbc.DropdownMenu(
    # children=[
    #     dbc.DropdownMenuItem("Home", href="/apps/page1"),
    #     dbc.DropdownMenuItem("Resultat logs", href="/apps/page2"),

    # ],
    # nav = True,
    # in_navbar = True,
    # label = "Explore",
    # )

# plot

fig = go.Figure(data=[
    go.Bar(name='frequence des emotions',
            x=grouped.index, 
            y= grouped[('Text', 'count')], 
            marker = dict(color = '#a1b5cc',
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
            marker = dict(color = '#a1b5cc',
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
    marker = dict(color = '#b2cca1',
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
    marker = dict(color = '#b2cca1',
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

fig6 = go.Figure(data=[
    go.Bar(name='frequence des emotions',
            x=grouped2.index, 
            y= grouped2[('Text', 'count')], 
            marker = dict(color = '#a1b5cc',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = grouped2[('Text', 'count')])
    ])
fig6.update_layout(barmode='group',title = 'Fréquence des emotions ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))

fig7 = go.Figure(data=[
    go.Bar(name= 'frequence des mots',
    x = r2,
    y = freq2, 
    marker = dict(color = '#b2cca1',
            line = dict(color ='rgb(0,0,0)',width =2.5)),
            text = subsample(labels2))
])
fig7.update_layout(barmode='group',title = 'Fréquence des mots ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))
fig7.update_xaxes(
        tickmode='array',
        tickvals = r2,
        ticktext = labels2)

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
                         'maxWidth': '1600px'},
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
            html.Br(),
            html.Br(),
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
                         'maxWidth': '1600px'},
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
            html.Br(),
            html.Br(),
            dcc.Graph(
                figure=fig1
                ),
            dcc.Graph(
                figure=fig3
                )
            ]),
        dcc.Tab(label='Data merger', children=[
            dash_table.DataTable(
            id='kaggle',
            columns=[{"name": i, "id": i} for i in df2.columns],
            data=df2.to_dict('records'),
            editable=False,
            css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}],
            style_table={'overflowX': 'scroll',
                         'overflowY': 'scroll',
                         'maxHeight': '300px',
                         'maxWidth': '1600px'},
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
            html.Br(),
            html.Br(),
            dcc.Graph(
                id = 'plot',
                figure= fig6
            ),
            dcc.Graph(
                id = 'plot2',
                figure= fig7
            )])
        
    ]),
    html.Br(),
    html.Br(),
    html.Div(id='id1'),
    dcc.Link('Go to page2', href='/apps/page2')
])

def print_table(res):
    # Compute mean 
    final = {}
    for model in res:
        arr = np.array(res[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,3].mean().round(3)
        }
    df3 = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df3

filename = 'filename.pkl'
with open(filename, 'rb') as f:
    printTable = print_table(pickle.load(f))

def print_table1(res1):
    # Compute mean 
    final = {}
    for model in res1:
        arr = np.array(res1[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,3].mean().round(3)
        }
    df4 = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df4

filename1 = 'filename1.pkl'
with open(filename1, 'rb') as f1:
    printTable1 = print_table1(pickle.load(f1))

def print_table2(res2):
    # Compute mean 
    final = {}
    for model in res2:
        arr = np.array(res2[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,3].mean().round(3)
        }
    df4 = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df4

filename2 = 'filename2.pkl'
with open(filename2, 'rb') as f2:
    printTable2 = print_table2(pickle.load(f2))


# plot f1_score recall and precision from data kaggle

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.time, name="time",
                    line_shape='linear'))
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.f1_score, name="f1_score",
                    line_shape='linear'))
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.Precision, name="Precision",
                    line_shape='linear'))
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.Recall, name="Recall",
                    line_shape='linear'))

fig4.update_traces(hoverinfo='text+name', mode='lines+markers')
fig4.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=printTable1.name, y=printTable1.time, name="time",
                    line_shape='linear'))
fig5.add_trace(go.Scatter(x=printTable1.name, y=printTable1.f1_score, name="f1_score",
                    line_shape='linear'))
fig5.add_trace(go.Scatter(x=printTable1.name, y=printTable1.Precision, name="Precision",
                    line_shape='linear'))
fig5.add_trace(go.Scatter(x=printTable1.name, y=printTable1.Recall, name="Recall",
                    line_shape='linear'))

fig5.update_traces(hoverinfo='text+name', mode='lines+markers')
fig5.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=printTable2.name, y=printTable2.time, name="time",
                    line_shape='linear'))
fig8.add_trace(go.Scatter(x=printTable2.name, y=printTable2.f1_score, name="f1_score",
                    line_shape='linear'))
fig8.add_trace(go.Scatter(x=printTable2.name, y=printTable2.Precision, name="Precision",
                    line_shape='linear'))
fig8.add_trace(go.Scatter(x=printTable2.name, y=printTable2.Recall, name="Recall",
                    line_shape='linear'))

fig8.update_traces(hoverinfo='text+name', mode='lines+markers')
fig8.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

layout2 = html.Div([
    Header(),
    html.Br(),
    html.Br(), 
    dcc.Tabs([
    dcc.Tab(label='Kaggle Data', children=[
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable.columns],
            data=printTable.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        dcc.Markdown('''On remarque qu'on as des scores très élever qui se rapproche des 90% 
                        qui veut dire qu'on as des bonnes prédictions'''),
        html.Br(),
        dcc.Graph(
            id = 'plot4',
            figure=fig4
        ),
        dcc.Markdown(''' Matrice de confusion'''),
        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix.png'), style = { 'width': '450px', 'height':'400px'})])
        #html.Div(children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image1))])

        ]),
    dcc.Tab(label='Data Word', children=[
        dash_table.DataTable(
            id = 'data W',
            columns=[{"name": i, "id": i} for i in printTable1.columns],
            data=printTable1.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        dcc.Markdown(''' On remarque qu'on as des scores très faible due au nuance des emortions
                        dans les messages
                        '''),
        html.Br(),
        dcc.Graph(
            figure= fig5
        ), 
        dcc.Markdown(''' Matrice de confusion'''),

        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix1.png'), style = { 'width': '450px', 'height':'400px'},)])
    ]),
    dcc.Tab(label='Data merge', children=[
        dash_table.DataTable(
            id = 'data m',
            columns=[{"name": i, "id": i} for i in printTable2.columns],
            data=printTable2.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        dcc.Markdown(''' On remarque qu'il y as une amélioration du score f1 cela du au bon score
                        de la prédiction du jeu de donnée de kaggle qui as amélioré le score 
                        du jeu de donnée data worldqu'on asqu'on as des scores très faible due au nuance des emortions
                        dans les messages des scores très faible due au nuance des emotions
                        dans les messages
                        '''),
        html.Br(),
        dcc.Graph(
            figure= fig8
        ), 
        dcc.Markdown(''' Matrice de confusion'''),
        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix2.png'), style = { 'width': '450px', 'height':'400px'},)])
        ])  
    ]), 
    html.Br(),
    html.Br(),
    html.Div(id='id2'),
    dcc.Link('Go to page1', href='/apps/page1')
])            

