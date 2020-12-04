import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
from app import app
import dash_table
from urllib.parse import quote
import plotly.graph_objs as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, preprocessing
import numpy as np
#from functions import *


colors = {
    'background': '#00000',
    'text': '#111111'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv("./data/timesData.csv")
df.international = pd.to_numeric(df.international, errors='coerce')


df.num_students  = [str(each).replace(',', '') for each in df.num_students]
df.num_students =  pd.to_numeric(df.num_students, errors='coerce')//1000
df.country = [str(each.replace('Unisted States of America', 'United States of America')) for each in df.country]
df.country = [str(each.replace('Austria', 'Australia')) for each in df.country]
df.country = [str(each.replace('unted Kingdom', 'United Kingdom')) for each in df.country]
international_color = [float(each) for each in df.international]
df.income = pd.to_numeric(df.income, errors='coerce')
df.total_score = pd.to_numeric(df.total_score, errors = 'coerce')
df.international = pd.to_numeric(df.international, errors = 'coerce')
df.world_rank = [str(each).replace('=','') for each in df.world_rank]
df = df[df.year == 2016]
df = df.iloc[:50,:]

n_comp = 8
X = df.values
data_pca = df[["teaching", "international","research", "citations","income", "total_score","num_students", "student_staff_ratio"]]

data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
X = data_pca.values
names = df["university_name"] # ou data.index pour avoir les intitulés
features = df.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = PCA(n_components=n_comp)
pca.fit(X_scaled)
#X = df[features]

# pca = PCA()
# components = pca.fit_transform(df[features])
# Eboulis des valeurs propres
#display_scree_plot(pca)
scree = pca.explained_variance_ratio_*100
# fig = px.bar(X_scaled, x=np.arange(len(scree))+1,
#              y=scree.cumsum())
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    X_scaled,
    labels=labels,
    dimensions=range(4),
    color=df["university_name"]
)
fig.update_traces(diagonal_visible=False)

html.A(html.Button('Download Data'),
        id="download-button",
        download='Data2016.csv',
        href="data:text/csv;charset=utf-8,"+quote(df.to_csv(index=False)),
        target="_blank"),
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
                'Les 50 meilleurs université')
        ], className="twelve columns padded")

    ], className="row")
    return header

def get_menu():
    menu = html.Div([

        dcc.Link('page 1     |', href='/apps/page1', className="tab first"),

        dcc.Link('page 2      |', href='/apps/page2', className="tab")

    ], className="row")
    return menu

html.Br(),
html.Br(),

layout1 = html.Div([
    Header(),
    html.Div([   
        dash_table.DataTable(
            id='table2016',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            editable=True,
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
    )]
    ),
    html.Br(),
    html.A(html.Button('Download Data'),
            id="download-button",
            download='Data2016.csv',
            href="data:text/csv;charset=utf-8,"+quote(df.to_csv(index=False)),
            target="_blank"),
    html.Br(),
    html.Br(),
    html.Div([  
        dcc.Graph(
            id='plot1',
            figure={
                'data': [
                    go.Scatter(
                        x=df[df['country'] == i]['world_rank'],
                        y=df[df['country'] == i]['research'],
                        text=df[df['country'] == i]['university_name'],
                        mode='markers',
                        opacity=0.8,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df.country.unique()
                ],
                'layout': go.Layout(
                    xaxis={'title': 'world_rank'},
                    yaxis={'title': 'research'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0.0, 'y': 1},
                    hovermode='closest'
            )
        }
    )]),
    html.Br(),
    html.Br(),
    html.Div(dcc.Markdown('''## Les enseignements par rapport au nom des universite''')),
    html.Div([
        # dcc.Input(
        #     value = 0,
        #     type = 'text'
        # ),
        dcc.Graph(id='plot2',
            
            figure={
                'data': [
                    {
                        'y': df.teaching,
                        'x':df.university_name,
                        'mode': 'markers',
                        'marker':{
                            'color':international_color,
                            'size': df.num_students,
                            'showscale': True,
                            'colorbar': dict(title = '')
                        }
                        
                        #  'text':df[['university_name', 'num_students']],
                        #  'hovertamplate': "<b>university :</b> %{text[0]}<br>" + "<b>Student_num :</b> %{text[1]}"
                    }      
                ]
                
                # 'layout': dict(title = "Impact à l'international en fonction du nombre d'élèves et du classement mondial universitaire"),
                #                 'xaxis' : dict(title = 'Rang Mondial',ticklen = 5,zeroline = False),
                #                 'yaxix' : dict(title = 'Enseignement',ticklen= 5,zeroline = False)
            }
        )
    ]),
    html.Br(),
    html.Br(),
    html.Div(id='id1'),
    dcc.Link('Go to page1', href='/apps/page2')
]),


layout2 = html.Div([
    Header(),
    html.Div([
    html.Div(dcc.Markdown('''## Matrice de correlation''')),
    dcc.Graph(
        id =  'plot3',
        figure = fig
    )
    #figure.update_traces(diagonal_visible=False)
    ]),
    html.Br(), 
    html.Br(),
    html.Div([  
    html.Div(dcc.Markdown('''## Eboulis des valeurs propres'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '34vw'},children = [html.Img(id='image',src=app.get_asset_url('Eboulis des valeurs propres.png'))]),
    html.Div(dcc.Markdown('''## cercle de correlation F1 F2'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('cercle de correlation F1 F2.png'))]),
    html.Div(dcc.Markdown('''## cercle de correlation F3 F4'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('cercle de correlation F3 F4.png'))]),
    html.Div(dcc.Markdown('''## cercle de correlation F5 F6'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('cercle de correlation F5 F6.png'))]),
    html.Div(dcc.Markdown('''## projection des individus F1 F2'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('projection des individus F1 F2.png'))]),
    html.Div(dcc.Markdown('''## projection des individus F3 F4.'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('projection des individus F3 F4.png'))]),
    html.Div(dcc.Markdown('''## projection des individus F5 F6'''), style = {'text-align' : 'center'}),
    html.Div(style = {'padding-left': '33vw'},children = [html.Img(id='image',src=app.get_asset_url('projection des individus F5 F6.png'))]),
    ]), 
    html.Br(),
    html.Br(),
    html.Div(id='id2'),
    dcc.Link('Go to page1', href='/apps/page1')
]),


@app.callback(
    Output("plot2", "figure"), 
    Input("idgraph", "value"))
def legend(pos_x, pos_y):
    fig = px.scatter(
    df, x="world_rank", y="teaching", 
    color="international_color", size="num_students", 
    size_max=45, log_x=True)
    fig.update_layout(legend_x=pos_x, legend_y=pos_y)
    return fig
