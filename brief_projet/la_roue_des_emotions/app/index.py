import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from app import server
from app import app
from layouts import layout1, layout2, layout3, layout4
import callbacks
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/apps/page1"),
        dbc.DropdownMenuItem("Data Vise", href="/apps/page2"),
        dbc.DropdownMenuItem("Resultat logs", href="/apps/page3"),
        dbc.DropdownMenuItem("Prediction", href="/apps/page4")


    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
    )

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/emotions.png", height="60px")),
                        dbc.Col(dbc.NavbarBrand("La Roue Des Emotions", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]        
    ),
        color="dark",
        dark=True,
        className="mb-4",
    )

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' or pathname == '/apps/page1':
        return layout1
    elif pathname == '/apps/page2':
        return layout2
    elif pathname == '/apps/page3':
        return layout3
    elif pathname == '/apps/page4':
        return layout4
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True) 