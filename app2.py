import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
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
from datetime import datetime, timedelta
import nltk 
from nltk import FreqDist
import copy

# Initiate app
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no"}]
)

githublink = 'https://github.com/s153748/extreme-weather-detection'
mapbox_access_token = open(".mapbox_token.txt").read()

# Load data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data") 
df = pd.read_csv(DATA_PATH.joinpath("final_tweets.csv")) 

# Data prep
total_count = len(df)
df['hashtags'] = [literal_eval(s) for s in df['hashtags']]
df['localization'] = df['localization'].astype(str)
df['date'] = pd.to_datetime(df['date'])

def unix_time(dt):
    return (dt-datetime.utcfromtimestamp(0)).total_seconds() 
init_start = unix_time(df['date'].min())
init_end = unix_time(df['date'].max())
init_start_date = datetime.utcfromtimestamp(init_start).strftime('%Y-%m-%d')
init_end_date = datetime.utcfromtimestamp(init_end).strftime('%Y-%m-%d')

# Set graph options
graph_list = ['Scatter map','Hexagon map']
style_list = ['Light','Dark','Streets','Outdoors','Satellite'] 
color_list = ['Localization','Retweeted']
loc_list = df.localization.unique()
retweet_list = df.retweeted.unique()
colors = ['#003f5c', '#ffa600', '#ef5675', '#7a5195']

# Create global chart template
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#171b26",
    paper_bgcolor="#171b26",
    hovermode="closest",
    #legend=dict(font=dict(size=12, color='#737a8d'), orientation="h"),
    mapbox=dict(accesstoken=mapbox_access_token,
                style='light',
                center=go.layout.mapbox.Center(lat=20, lon=-3),
                zoom=0.65,
    )
)

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
                            html.Label("Filter by"),
                            dcc.Dropdown(
                            #dcc.RadioItems(
                                id="color-select",
                                options=[{"label": i, "value": i} for i in color_list],
                                value=color_list[0],
                                #labelStyle={"display": "inline-block"},
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
                            html.Label("Localization Method"),
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
                                style={'width':'100%','height':'1px','background-color':'#171b26','opacity':0.5,'color':'#ffffff'},
                                draggable=False,
                                placeholder='e.g. Floods, Queensland'
                            ),
                            html.Button('Search', id='search-button', n_clicks=0),
                            html.Div(id='counter',style={'color':'#7b7d8d','fontsize':'9px','margin-top':'20px'}),
                        ]
                    )
                ]
           )
        ]
    )

def filter_data(df, range_select, loc_select, n_clicks, keywords):
        
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            df = df[df['full_text'].str.contains(keyword, case=False)]
    df = df[df['localization'].isin(loc_select)]
    df = df[df['date'] >= range_select[0]]
    filtered_df = df[df['date'] <= range_select[1]]
    
    return filtered_df

def generate_barchart(df, range_select, loc_select, n_clicks, keywords):
    
    graph_layout = copy.deepcopy(layout)
    filtered_df = filter_data(df, [init_start_date,init_end_date], loc_select, n_clicks, keywords)
    
    g = filtered_df[['date']]
    g.index = g['date']
    g = g.resample('D').count()
    g.rename(columns={'date':'count'},inplace=True)
    
    cols = []
    for i in g.index:
        if i.strftime('%Y-%m-%d') >= range_select[0] and i.strftime('%Y-%m-%d') <= range_select[1]:
            cols.append("rgb(197, 221, 240)")
        else:
            cols.append("rgba(197, 221, 240, 0.2)")
    
    data = [
        dict(
            type="scatter",
            mode="markers",
            x=g.index,
            y=g["count"] / 2,
            name="Count",
            opacity=0,
            hoverinfo="skip",
        ),
        dict(
            type="bar",
            x=g.index,
            y=g["count"],
            name="Count",
            marker=dict(color=cols),
            hovertemplate ='<b>%{x} </b><br>Count: %{y}'
        ),
    ]
    graph_layout["dragmode"] = "select"
    graph_layout["showlegend"] = False
    graph_layout["height"] = 100
    
    fig = dict(data=data, layout=graph_layout)
    
    return fig

def generate_geo_map(geo_df, graph_select, style_select, color_select, graph_layout):
    
    if graph_select == 'Scatter map':
        traces = []
        i = 0
        for color_type, dff in geo_df.groupby(color_select):
            tweet = dff["full_text"]
            localization = dff["localization"]
            user_name = dff['user_name']
            created_at = dff['created_at']
            source = dff['source']
            retweet_count = dff['retweet_count']
            
            trace = dict(
                type="scattermapbox",
                lat=dff["lat"],
                lon=dff["lon"],
                name=loc_list[i] if color_select=='localization' else retweet_list[i],
                customdata=tweet,
                hoverinfo="text",
                text='<b>'+tweet+'</b><br>localization: '+localization+'<br>user_name: '+user_name+'<br>created_at: '+created_at+'<br>source: '+source+'<br>retweet_count: '+retweet_count+'<br>',
                marker=dict(size=4, opacity=0.7, color=colors[i]),
            )
            traces.append(trace)
            i += 1
        
        if graph_layout is not None:
            if "mapbox.center" in graph_layout.keys():
                layout["mapbox"]["center"]["lon"] = float(graph_layout["mapbox.center"]["lon"])
                layout["mapbox"]["center"]["lat"] =  float(graph_layout["mapbox.center"]["lat"])
                layout["mapbox"]["zoom"] = float(graph_layout["mapbox.zoom"])
        layout["mapbox"]["style"] = style_select
        
        fig = dict(data=traces, layout=layout)
        
    elif graph_select == 'Hexagon map':
        fig = ff.create_hexbin_mapbox(geo_df, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=int(max(25,len(geo_df)/20)), 
                                      opacity=0.7, 
                                      labels={"color": "Count"},
                                      min_count=1, 
                                      color_continuous_scale='GnBu',
        )
        
    return fig

def generate_treemap(filtered_df):
    k = 20
    hashtag_list = [hashtag for sublist in filtered_df['hashtags'].tolist() for hashtag in sublist]
    freq_df = pd.DataFrame(list(FreqDist(hashtag_list).items()), columns = ["hashtag","count"]) 
    freq_df = freq_df.sort_values('count',ascending=False)
    fig = go.Figure(go.Treemap(
                        labels=freq_df['hashtag'][:k].tolist(),
                        values=freq_df['count'][:k].tolist(),
                        parents=['']*k,
                        marker_colorscale=px.colors.sequential.Teal,
                        hovertemplate='<b>%{label} </b> <br>Count: %{value}<extra></extra>'))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=20), 
                      height=240,
                      plot_bgcolor="#171b26",
                      paper_bgcolor="#171b26") 
    return fig

def generate_table(filtered_df):
    text_df = filtered_df[['full_text']]
    text_df.rename(columns={'full_text':'Tweets'},inplace=True)
    table = dash_table.DataTable( 
        id="tweets-table",
        columns=[{"name": i, "id": i} for i in text_df.columns],
        data=text_df.to_dict('records'),
        page_size=5,
        style_cell={'whiteSpace':'normal','height':'auto','width':'250px',"background-color":"#242a3b","color":"#7b7d8d"},
        style_as_list_view=False,
        style_header={"background-color":"#1f2536",'fontWeight':'bold',"padding":"0px 5px"},
    )
    return table

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.Div(
                id="banner",
                className="banner mb-4",
                children=[
                    html.H6("Extreme Weather Event Detection"),
                    html.A([
                        html.Img(
                            src=app.get_asset_url("GitHub-Mark-Light-64px.png")
                        )
                    ], href=githublink)
                ],
            ), 
            xs=12, sm=12, md=12, lg=12, xl=12
        )
    ),
    dbc.Row([
        dbc.Col([
            html.Div(id="output-clientside"), 
            build_control_panel(),
        ], 
            xs=12, sm=12, md=2, lg=2, xl=2
        ),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.P(
                        id="map-title",
                        children="Spatio-Temporal Development of Flood-Relevant Tweets"
                    ),
                    html.Div(
                        id="geo-map-outer",
                        children=[
                            dcc.Loading(children=dcc.Graph(
                                id="geo-map",
                            )),
                        ],
                    ),
                    html.Div([
                        dcc.RangeSlider(
                            id='range-slider',
                            min=init_start,
                            max=init_end, 
                            value=[init_start, init_end], 
                            updatemode='mouseup',
                        ), 
                    ], style={'height':'20px','margin-bottom':'1px'}),
                    html.Div(id='output-range-slider',style={'color':'#7b7d8d','fontsize':'8px','margin-bottom':'1px'}),
                    html.Div([
                        dcc.Loading(children=dcc.Graph(
                            id="barchart",
                        )),
                    ]),
                ], 
                    xs=12, sm=12, md=9, lg=9, xl=9
                ),
                dbc.Col([
                    html.Div(
                        children=[
                            html.P(
                                id="treemap-title",
                                children="Content"
                            ),
                            html.Div(
                                id="treemap-outer",
                                children=[
                                    dcc.Loading(children=dcc.Graph(
                                        id='treemap',
                                        figure={
                                            "data": [], "layout": dict(plot_bgcolor="#171b26", paper_bgcolor="#171b26"),
                                        },
                                    )),
                                ],
                            ),
                            html.Div(
                                id="tweets-outer",
                                children=[
                                    dcc.Loading(children=html.Div(id="tweets-table")),
                                ],
                            ),
                        ],
                    ),
                ], 
                    xs=12, sm=12, md=3, lg=3, xl=3
                ),
            ])
        ], 
            xs=12, sm=12, md=10, lg=10, xl=10
        )
    ], no_gutters=False, justify='start')
], fluid=True)

# Slider -> Barchart
@app.callback(
    Output("range-slider", "value"),
    [
        Input("barchart", "selectedData")
    ]
)
def update_slider(barchart_selected):
    if barchart_selected is None:
        return [init_start, init_end]
    
    nums = [int(point["pointNumber"]) for point in barchart_selected["points"]]
    start = unix_time(datetime.strptime('2013-01-27','%Y-%m-%d') + timedelta(days=min(nums)))
    end = unix_time(datetime.strptime('2013-01-28','%Y-%m-%d') + timedelta(days=max(nums)))

    return [start, end]

# Selectors -> Map, Barchart, Table, Treemap
@app.callback(
    [
        Output('barchart', 'figure'),
        Output('geo-map', 'figure'),
        Output('treemap', 'figure'),
        Output('tweets-table', 'children'),
        Output('counter', 'children'),
        Output('output-range-slider', 'children'),
    ], [
        Input('range-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('color-select', 'value'),
        Input('loc-select', 'value'),
        Input('search-button', 'n_clicks')
    ], [
        State('text-search', 'value'),
        State('geo-map', 'relayoutData')
    ], 
)
def update_visuals(range_select, graph_select, style_select, color_select, loc_select, n_clicks, keywords, graph_layout):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    filtered_df = filter_data(df, [start,end], loc_select, n_clicks, keywords)
    barchart = generate_barchart(df, [start,end], loc_select, n_clicks, keywords)
    geomap = generate_geo_map(filtered_df, graph_select, style_select.lower(), color_select.lower(), graph_layout)
    treemap = generate_treemap(filtered_df)
    table = generate_table(filtered_df)
    pct = np.round(len(filtered_df)/total_count*100,1)
    counter = f'Tweets in selection: {len(filtered_df)} ({pct}%)' if pct < 100 else f'Tweets in selection: {len(filtered_df)}'
    period = f'Selected period: {pd.to_datetime(start).strftime("%b %d, %Y")} - {pd.to_datetime(end).strftime("%b %d, %Y")}'

    return barchart, geomap, treemap, table, counter, period

if __name__ == '__main__':
    app.run_server()
