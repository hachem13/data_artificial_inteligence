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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from  sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import dash_bootstrap_components as dbc
from time import time
from collections import defaultdict
import pickle
import nltk

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

html.Br(),

def Header():
    return html.Div([
        html.Br([]),
        html.Br([]),
        html.Br([]),

    ]) 
 
layout1 = html.Div([html.Div(style = {'padding-left': '40vw'},children = [html.Img(id='image',src=app.get_asset_url('emotions2.png'))]),
        html.Br(),
        html.Br(),
        html.Div(style = {'text-size': '40px', 'text_aligne': 'center'}, children = [dcc.Markdown('''Construit d’après les travaux du psychologue américain Robert Plutchik, la roue des émotions est un modèle des émotions humaines et peut facilement servir à définir des personnages, ainsi que leur évolution dans une trame narrative.

Depuis quelques années, les dispositifs de communication médiatisée par ordinateur (CMO) sont massivement utilisés, aussi bien dans les activités professionnelles que personnelles. Ces dispositifs permettent à des participants distants physiquement de communiquer. La plupart implique une communication écrite médiatisée par ordinateur (CEMO) : forums de discussion, courrier électronique, messagerie instantanée. Les participants ne s’entendent pas et ne se voient pas mais peuvent communiquer par l’envoi de messages écrits, qui combinent, généralement, certaines caractéristiques des registres écrit et oral (Marcoccia, 2000a ; Marcoccia, Gauducheau, 2007 ; Riva, 2001).

Imaginez que vous souhaitez savoir ce qui se passe derrière votre écran ordinateur, qui sont vos contacts les plus actifs et quelle est leur personnalité (pas banal comme question !!). Vous allez alors vous lancer dans l’analyse de leur narration et tenter d’extraire quelle émotion se dégage de chacune des phrases.

Chez Simplon nous utilisons tous les jours des outils de discussion textuels et nous construisons nos relations sociales et professionnelles autour de ces dispositifs. Pour entretenir des rapports sociaux stables, sereins, de confiance et efficaces, au travers des outils de communication écrites, lorsqu'il n'est pas possible d'avoir la visio (avec caméra), il est nécessaire de détecter des éléments "clés" dans les channels de discussions / mails qui nous permettront de déceler de la colère, de la frustration, de la tristesse ou encore de la joie de la part d'un collègue ou d'un amis pour adapter nos relations sociales.
En tant qu'expert en data science, nous allons vous demander de développer un modèle de machine learning permettant de classer les phrases suivant l'émotion principale qui en ressort.

Pour des questions d’ordre privé, nous ne vous demanderons pas de nous communiquer les conversations provenant de votre réseau social favori ou de vos emails mais nous allons plutôt vous proposer deux jeux de données contenant des phrases, ces fichiers ayant déjà été annoté.

Vous devrez proposer plusieurs modèles de classification des émotions et proposer une analyse qualitative et quantitative de ces modèles en fonction de critères d'évaluation. Vous pourrez notamment vous appuyer sur les outils de reporting des librairies déjà étudiées. Vous devrez investiguer aux travers de librairies d'apprentissage automatique standards et de traitement automatique du langage naturel comment obtenir les meilleurs performance de prédiction possible en prenant en compte l'aspect multi-class du problème et en explorant l'impact sur la prédiction de divers prétraitement tel que la suppression des **stop-words**, la **lemmatisation** et l'utilisation de **n-grams**, et différente approche pour la vectorisation.

Vous devrez travailler dans **un premier temps** avec le jeu de données issue de [**Kaggle**](https://www.kaggle.com/ishantjuyal/emotions-in-text) pour réaliser vos apprentissage et l'évaluation de vos modèles.

Dans l'objectif d'enrichir notre prédictions nous souhaitons augmenter notre jeux de donneés.
Vous devrez donc travailler dans un **deuxième temps** avec le jeux de données fournie, issue de [**data.world**](https://data.world/crowdflower/sentiment-analysis-in-text) afin de  :
1. comparer d'une part si les résultats de classification sur votre premier jeux de données sont similaire avec le second. Commentez.
2. Combiner les deux jeux données pour tenter d'améliorer vos résultats de prédiction.
3. Prédire les nouvelles émotions présente dans ce jeux de données sur les message du premier, et observer si les résultats sont pertinent.


Vous devrez ensuite présenter vos résultats sous la forme d'un dashboard muli-pages Dash.
La première page du Dashboard sera dédiée à l'analyse et au traitement des données. Vous pourrez par exemple présenter les données "brut" sous la forme d'un tableau puis les données pré-traitées dans le même tableau avec un bouton ou menu déroulant permettant de passer d'un type de données à un autre (n'afficher qu'un échantillon des résultats, on dans une fenetre "scrollable"). Sur cette première page de dashboard seront accessibles vos graphiques ayant trait à votre première analyse de données (histogramme, bubble chart, scatterplot etc), notamment
* l'histogramme représentant la fréquence d’apparition des mots (commentez)
* l'histogramme des émotions (commentez)

Une deuxième page du Dashboard sera dédiée aux résultats issues des classifications . Il vous est demandé de comparer les résultats d'au moins 5 classifiers qu présenterai dans un tableau permettant de visualiser vos mesures. Sur cette page de dashboard pourra se trouver par exemple, des courbes de rappel de précision (permette de tracer la précision et le rappel pour différents seuils de probabilité), un rapport de classification (un rapport de classification visuel qui affiche la precision, le recall, le f1-score, support, ou encore une matrice de confusion ou encore une graphique permettant de visualiser les mots les plus représentatif associé à chaque émotions.

 Héberger le dashboard sur le cloud de visualisation de données Héroku (https://www.heroku.com/)


**BONUS**

Créer une application client/serveur permettant à un utilisateur d'envoyer du texte via un champs de recherche (ou un fichier sur le disque du client) et de lui renvoyer
1. l'émotion du texte envoyé.
2. (bonus du bonus) la roue des émotions du document (exemple: quelle proportion de chacune des émotions contient le document ?)
'''
)])])


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

layout2=  html.Div([
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
    html.Div(id='id2'),
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

filename1 = 'filename1.pkl'
with open(filename1, 'rb') as f1:
    printTable1 = print_table(pickle.load(f1))

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

filename2 = 'filename2.pkl'
with open(filename2, 'rb') as f2:
    printTable2 = print_table1(pickle.load(f2))

filename3 = 'filename3.pkl'
with open(filename3, 'rb') as f3:
    printTable3 = print_table1(pickle.load(f3))


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

filename4 = 'filename4.pkl'
with open(filename4, 'rb') as f4:
    printTable4 = print_table2(pickle.load(f4))

filename5 = 'filename5.pkl'
with open(filename5, 'rb') as f5:
    printTable5 = print_table2(pickle.load(f5))

# plot f1_score recall and precision from data kaggle

fig4 = go.Figure()

fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.f1_score, name="f1_score",
                    line_shape='linear'))
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.Precision, name="Precision",
                    line_shape='linear'))
fig4.add_trace(go.Scatter(x=printTable.name, y=printTable.Recall, name="Recall",
                    line_shape='linear'))

fig4.update_traces(hoverinfo='text+name', mode='lines+markers')
fig4.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

fig9 = go.Figure()

fig9.add_trace(go.Scatter(x=printTable1.name, y=printTable1.f1_score, name="f1_score",
                    line_shape='linear'))
fig9.add_trace(go.Scatter(x=printTable1.name, y=printTable1.Precision, name="Precision",
                    line_shape='linear'))
fig9.add_trace(go.Scatter(x=printTable1.name, y=printTable1.Recall, name="Recall",
                    line_shape='linear'))

fig9.update_traces(hoverinfo='text+name', mode='lines+markers')
fig9.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall avec tfid')

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=printTable2.name, y=printTable2.f1_score, name="f1_score",
                    line_shape='linear'))
fig5.add_trace(go.Scatter(x=printTable2.name, y=printTable2.Precision, name="Precision",
                    line_shape='linear'))
fig5.add_trace(go.Scatter(x=printTable2.name, y=printTable2.Recall, name="Recall",
                    line_shape='linear'))

fig5.update_traces(hoverinfo='text+name', mode='lines+markers')
fig5.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

fig10 = go.Figure()
fig10.add_trace(go.Scatter(x=printTable3.name, y=printTable3.f1_score, name="f1_score",
                    line_shape='linear'))
fig10.add_trace(go.Scatter(x=printTable3.name, y=printTable3.Precision, name="Precision",
                    line_shape='linear'))
fig10.add_trace(go.Scatter(x=printTable3.name, y=printTable3.Recall, name="Recall",
                    line_shape='linear'))

fig10.update_traces(hoverinfo='text+name', mode='lines+markers')
fig10.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall avec tfid' )

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=printTable4.name, y=printTable4.f1_score, name="f1_score",
                    line_shape='linear'))
fig8.add_trace(go.Scatter(x=printTable4.name, y=printTable4.Precision, name="Precision",
                    line_shape='linear'))
fig8.add_trace(go.Scatter(x=printTable4.name, y=printTable4.Recall, name="Recall",
                    line_shape='linear'))

fig8.update_traces(hoverinfo='text+name', mode='lines+markers')
fig8.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')

fig11 = go.Figure()
fig11.add_trace(go.Scatter(x=printTable5.name, y=printTable5.f1_score, name="f1_score",
                    line_shape='linear'))
fig11.add_trace(go.Scatter(x=printTable5.name, y=printTable5.Precision, name="Precision",
                    line_shape='linear'))
fig11.add_trace(go.Scatter(x=printTable5.name, y=printTable5.Recall, name="Recall",
                    line_shape='linear'))

fig11.update_traces(hoverinfo='text+name', mode='lines+markers')
fig11.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16),title='f1_socre, precision, recall')
f

layout3 = html.Div([
    Header(),
    html.Br(),
    html.Br(), 
    dcc.Tabs([
    dcc.Tab(label='Kaggle Data', children=[
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos sans tfid'''),
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable.columns],
            data=printTable.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos avec tfid'''),
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable1.columns],
            data=printTable1.to_dict('records'),
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
        dcc.Graph(
            id = 'plot9',
            figure=fig9
        ),
        dcc.Markdown(''' Matrice de confusion'''),
        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix.png'), style = { 'width': '450px', 'height':'400px'})])
        #html.Div(children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image1))])

        ]),
    dcc.Tab(label='Data Word', children=[
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos sans tfid'''),
        dash_table.DataTable(
            id = 'data W',
            columns=[{"name": i, "id": i} for i in printTable2.columns],
            data=printTable2.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos avec tfid'''),
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable3.columns],
            data=printTable3.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        dcc.Markdown(''' On remarque qu'on as des scores très faible due au nuance des emortions
                        dans les messages
                        '''),
        html.Br(),
        dcc.Graph(
            figure= fig5
        ), 
        dcc.Graph(
            figure= fig10
        ), 
        dcc.Markdown(''' Matrice de confusion'''),

        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix1.png'), style = { 'width': '450px', 'height':'400px'},)])
    ]),
    dcc.Tab(label='Data merge', children=[
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos sans tfid'''),
        dash_table.DataTable(
            id = 'data m',
            columns=[{"name": i, "id": i} for i in printTable4.columns],
            data=printTable4.to_dict('records'),
            editable=False,
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
            ),
        html.Br(),
        html.Br(),
        dcc.Markdown('''Tableau des résultats des algos avec tfid'''),
        dash_table.DataTable(
            id = 'Kaggle1',
            columns=[{"name": i, "id": i} for i in printTable5.columns],
            data=printTable5.to_dict('records'),
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
        dcc.Graph(
            figure= fig11
        ), 
        dcc.Markdown(''' Matrice de confusion'''),
        html.Div(children = [html.Img(src=app.get_asset_url('confusion-matrix2.png'), style = { 'width': '450px', 'height':'400px'},)])
        ])  
    ]), 
    html.Br(),
    html.Br(),
    html.Div(id='id3'),
    dcc.Link('Go to page1', href='/apps/page1')
])            

stopwords = nltk.corpus.stopwords.words('english')


targets = list(df["Emotion"])
corpus = list(df["Text"])

X = corpus
y = targets
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

pipe = Pipeline([('vect', CountVectorizer(stop_words = stopwords)), ('sgd', SGDClassifier()),])
pipe.fit(X_train, y_train)




layout4 = html.Div(
    [
        html.I("Cette Page nous donne la prédiction des émotions sur le jeu de données Kaggle "),
        html.Br(),
        html.Br(),
        html.Div(dcc.Input(id="input1", type="text", placeholder="Ecrivez votre text", debounce=True), className="mb-4 text-center"),
        html.Div(html.H2(id="output"), className="mb-4 text-center"),
    ]
)