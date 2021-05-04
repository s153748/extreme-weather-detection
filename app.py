import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import ast 
from ast import literal_eval
import pandas as pd
import numpy as np
import pathlib
import calendar
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
import nltk 
from nltk import FreqDist

# Initiate app
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no"}]
)
server = app.server
app.config.suppress_callback_exceptions = True

githublink = 'https://github.com/s153748/extreme-weather-detection'
mapbox_access_token = open(".mapbox_token.txt").read()

# Load data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data") 
df = pd.read_csv(DATA_PATH.joinpath("final_tweets.csv")) 

# Data prep
total_count = len(df)
df['date'] = pd.to_datetime(df['date'])
df['hashtags'] = [literal_eval(s) for s in df['hashtags']]
df['localization'] = df['localization'].astype(str)

# Set graph options
graph_list = ['Scatter map','Hexagon map']
style_list = ['Light','Dark','Streets','Outdoors','Satellite'] 
color_list = ['Localization','Retweeted']
loc_list = df.localization.unique()
colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

def unix_time(dt):
    return (dt-datetime.utcfromtimestamp(0)).total_seconds() 

def get_marks(start, end):
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return {int(unix_time(m)): (str(m.strftime('%b %Y'))) for m in result}

def build_control_panel():
    return html.Div(
        id="control-panel",
        children=[
            html.P(
                className="section-title",
                children="Configurations",
            ),
            html.Div(
                className="control-row-1",
                children=[
                    html.Div(
                        id="graph-select-outer",
                        children=[
                            html.Label("Graph Type"),
                            dcc.Dropdown(
                                id="graph-select",
                                options=[{"label": i, "value": i} for i in graph_list],
                                value=graph_list[0],
                            ),
                        ],
                    )
                ]
            ),
            html.Div(
                className="control-row-2",
                children=[
                    html.Div(
                        id="style-select-outer",
                        children=[
                            html.Label("Map Style"),
                            dcc.Dropdown(
                                id="style-select",
                                options=[{"label": i, "value": i} for i in style_list],
                                value=style_list[0],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="control-row-3",
                children=[
                    html.Div(
                        id="color-select-outer",
                        children=[
                            html.Label("Color by"),
                            dcc.Dropdown(
                                id="color-select",
                                options=[{"label": i, "value": i} for i in color_list],
                                value=color_list[0],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="control-row-4",
                children=[
                    html.Div(
                        id="loc-select-outer",
                        children=[
                            html.Label("Localization"),
                            dcc.Dropdown(
                                id='loc-select',
                                options=[{"label": i, "value": i} for i in loc_list],
                                multi=True,
                                value=loc_list,
                                placeholder=''
                            )
                        ]
                    )
                ],
            ),
            html.Div(
                className="control-row-5",
                children=[
                    html.Div(
                        id="text-search-outer",
                        children=[
                            html.Label("Keywords"),
                            dcc.Textarea(
                                id='text-search',
                                value='',
                                style={'width': '100%', 'height': "1 px", 'background-color': '#171b26', 'opacity': 0.5, 'color': '#ffffff'},
                                draggable=False,
                                placeholder='e.g. Floods, Queensland'
                            ),
                            html.Button('Search', id='search-button', n_clicks=0),
                        ]
                    )
                ]
           ),
           html.Br(),
           html.Div(f'Total number of Tweets: {total_count}',style={'color':'#7b7d8d'}),
           html.Div(id='counter',style={'color':'#7b7d8d'}),
           html.Div(id='output-range-slider',style={'color':'#7b7d8d'})
        ],
    )

def generate_geo_map(geo_df, range_select, graph_select, style_select, color_select, loc_select, n_clicks, keywords):
    
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            geo_df = geo_df[geo_df['full_text'].str.contains(keyword, case=False)]
    
    geo_df = geo_df[geo_df['localization'].isin(loc_select)]
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    geo_df = geo_df[geo_df['date'] >= start]
    geo_df = geo_df[geo_df['date'] <= end]
    
    if len(geo_df) == 0: # no matches
        empty=pd.DataFrame([0, 0]).T
        empty.columns=['lat','long']
        fig = px.scatter_mapbox(empty, lat="lat", lon="long", color_discrete_sequence=['#cbd2d3'], opacity=0)
    
    elif graph_select == 'Scatter map':
        fig = px.scatter_mapbox(geo_df, 
                                lat="lat", 
                                lon="lon",
                                color=color_select.lower(),
                                hover_name='full_text',
                                hover_data={'lat':False,'lon':False,'localization':True,'user_location':True,'user_name':True,'created_at':True,'source':True,'retweet_count':True},
                                color_discrete_sequence=colors)

    elif graph_select == 'Hexagon map':
        fig = ff.create_hexbin_mapbox(geo_df, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=60, # int(max(25,len(geo_df)/10)), 
                                      opacity=0.7, 
                                      labels={"color": "Count"},
                                      min_count=1, 
                                      color_continuous_scale='GnBu')

    fig.update_layout(
        margin=dict(l=5, r=0, t=0, b=0),
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        mapbox=go.layout.Mapbox(accesstoken=mapbox_access_token,
                                center=go.layout.mapbox.Center(lat=40.4168, lon=-3.7037),
                                zoom=0.6,
                                style=style_select),
        font=dict(color='#737a8d'))
        
    return fig, geo_df, start, end

def generate_histogram(filtered_df, start, end):
    
    count_dates = filtered_df.groupby('date').size().values
    time_df = filtered_df.drop_duplicates(subset='date').assign(count=count_dates).sort_values(by='date').reset_index(drop=True)
    fig = px.histogram(time_df, 
                       x="date", 
                       y="count",
                       range_x=[start,end],
                       nbins=len(time_df)**2,
                       height=100,
                       color_discrete_sequence=['#cbd2d3'],)
    fig.update_traces(hovertemplate ='<b>%{x} </b><br>Count: %{y}') 
    fig.update_xaxes(showgrid=False,title='Date',tickformat="%d %b")
    fig.update_yaxes(showgrid=False,title='Count')
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), 
                      bargap=0.05,
                      plot_bgcolor="#171b26",
                      paper_bgcolor="#171b26",
                      font=dict(color='#7b7d8d',size=10)) # '#737a8d'
    return fig

def generate_treemap(filtered_df):
    
    k = 20
    hashtag_list = [hashtag for sublist in filtered_df['hashtags'].tolist() for hashtag in sublist]
    freq_df = pd.DataFrame(list(FreqDist(hashtag_list).items()), columns = ["hashtag","occurrence"]) 
    freq_df = freq_df.sort_values('occurrence',ascending=False)
    fig = go.Figure(go.Treemap(
                        labels=freq_df['hashtag'][:k].tolist(),
                        values=freq_df['occurrence'][:k].tolist(),
                        parents=['']*k,
                        marker_colorscale=px.colors.sequential.Teal,
                        hovertemplate='<b>%{label} </b> <br>Occurrences: %{value}<extra></extra>',
                   )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    
    return fig

# Set up the layout
app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Extreme Weather Event Detection"),
                html.A([
                    html.Img(
                        src=app.get_asset_url("GitHub-Mark-Light-64px.png")
                    )
                ], href=githublink)
            ]
        ),
        html.Div(
            id="left-column",
            className="three columns",
            children=[
                build_control_panel()
            ]
        ),
        html.Div(
            id="center-column",
            className="six columns",
            children=[
                html.Br(),
                html.P(
                    id="map-title",
                    children="Spatio-Temporal Development of Flood-Relevant Tweets"
                ),
                html.Div( 
                    id="geo-map-outer",
                    children=[
                        dcc.Graph(
                            id="geo-map",
                            figure={
                                "data": [], "layout": dict(plot_bgcolor="#171b26",paper_bgcolor="#171b26"),
                            },
                        ),
                        dcc.RangeSlider(
                            id='range-slider',
                            min=unix_time(df['date'].min()),
                            max=unix_time(df['date'].max())+1202400,
                            value=[unix_time(df['date'].min()), unix_time(df['date'].max())+1202400],
                            marks=get_marks(df['date'].min(),df['date'].max()),
                            updatemode='mouseup',
                        ),
                        dcc.Graph(
                            id="histogram",
                            figure={
                                "data": [], "layout": dict(plot_bgcolor="#171b26", paper_bgcolor="#171b26"),
                            },
                        )
                    ]
                ),
            ]
        ),
        html.Br(),
        html.Div(
            id="right-column",
            className="six columns",
            children=[
                html.Br(),
                html.P(
                    id="treemap-title",
                    children="Top hashtags"
                ),
                html.Div( 
                    id="treemap-outer",
                    children=[
                        dcc.Graph(
                            id='treemap',
                            figure={
                                "data": [], "layout": dict(plot_bgcolor="#171b26", paper_bgcolor="#171b26"),
                            },
                        )
                    ]
                ),
            ]
        )
    ]
)

@app.callback(
    [
        Output('geo-map', 'figure'),
        Output('histogram', 'figure'),
        Output('treemap', 'figure'),
        Output('output-range-slider', 'children'),
        Output('counter', 'children')
    ], [
        Input('range-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('color-select', 'value'),
        Input('loc-select', 'value'),
        Input('search-button', 'n_clicks'),
        State('text-search', 'value')
    ],
)
def update_visualizations(range_select, graph_select, style_select, color_select, loc_select, n_clicks, keywords):
    
    geo_map, filtered_df, start, end = generate_geo_map(df, range_select, graph_select, style_select.lower(), color_select, loc_select, n_clicks, keywords)
    line_chart = generate_histogram(filtered_df, start, end)
    treemap = generate_treemap(filtered_df)
    
    return geo_map, line_chart, treemap, f'Period: {start} - {end}', f'Tweets in selection: {len(filtered_df)}'

if __name__ == '__main__':
    app.run_server()
