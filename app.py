import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_dangerously_set_inner_html
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import pathlib
import calendar
import datetime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
df = pd.read_json(DATA_PATH.joinpath("eng_tweets_1718.json"),orient='split')
total_count = len(df)

def unix_time(dt):
    return (dt-datetime.utcfromtimestamp(0)).total_seconds() 
init_start = unix_time(df['date'].min())
init_end = unix_time(df['date'].max())
init_start_date = datetime.utcfromtimestamp(init_start).strftime('%Y-%m-%d')
init_end_date = datetime.utcfromtimestamp(init_end).strftime('%Y-%m-%d')
#years = df.year.unique()

def get_marks(start, end):
    result = []
    current = start
    while current <= end: 
        result.append(current)
        current += relativedelta(months=1) # 1 
    return {int(unix_time(m)): (str(m.strftime('%b %Y'))) for m in result} 

# Set graph options
graph_options = ['Scatter','Density','Hexabin']
style_options = ['light','dark','streets','satellite'] 
loc_options = ['Geoparsed from Tweet','Geotagged coordinates','Geotagged place','Registered user location']
type_options = ['Tweet','Retweet']
class_options = ['Logistic regression','Random forest','CNN','ULMFiT']
colors = ['#ef5675','#ffa600','#8073ac','#35978f']

# Create global chart template
layout = dict(
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#171b26",
    paper_bgcolor="#171b26",
    hovermode="closest",
    clickmode="event+select",
    mapbox=dict(accesstoken=mapbox_access_token,
                style='light',
                center=go.layout.mapbox.Center(lat=5, lon=5),
                zoom=1,
    ),
    legend=dict(bgcolor="rgba(203,210,211,0.2)",
                orientation="h",
                font=dict(color="#7b7d8d",size='8px'),
                x=0.02,
                y=0,
                yanchor="bottom",
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
                className="control-rows",
                children=[
                    html.Div(
                        id="graph-select-outer",
                        children=[
                            html.Label("Map Style"),
                            dcc.Dropdown(
                                id="graph-select",
                                options=[{"label": i, "value": i} for i in graph_options],
                                value=graph_options[0],
                            ),
                        ],
                    ),
                    html.Div(
                        id='style-select-outer',
                        children=[
                            dcc.Dropdown(
                                id="style-select",
                                options=[{"label": i.capitalize(), "value": i} for i in style_options],
                                value=style_options[1],
                            ),
                        ], style={'margin-top':'4px'}
                    ),
                    html.Div(
                        id="tweet-select-outer",
                        children=[
                            html.Label("Tweet Type"),
                            dcc.Dropdown(
                                id='type-select',
                                options=[{'label': i, 'value': i} for i in type_options],
                                value=type_options,
                                multi=True
                            ), 
                        ], style={'margin-top':'6px'}
                    ),
                    html.Div(
                         id="loc-select-outer",
                         children=[
                             html.Label("Location Type"),
                             dcc.Dropdown(
                                 id="loc-select",
                                 options=[{'label': i, 'value': i} for i in loc_options],
                                 value=loc_options[:3],
                                 multi=True
                             ),
                         ], style={'margin-top':'6px'}
                    ),
                    html.Div(
                         id="class-select-outer",
                         children=[ 
                            html.Label("Classifier"),
                            dcc.Dropdown(
                                id='class-select',
                                options=[{'label': i, 'value': i} for i in class_options],
                                value=class_options[3]
                            ),
                        ], style={'margin-top':'6px'}
                    ),
                    html.Div(
                        id="text-search-outer",
                        children=[
                            html.Label("Keywords"),
                            dcc.Textarea(
                                id='text-search',
                                value='',
                                style={'width':'100%','background-color':'#171b26','opacity':0.5,'color':'#ffffff'}, 
                                draggable=False,
                                placeholder='e.g. floods, #water'
                            ),
                            html.Button('Search', id='search-button', n_clicks=0),
                        ], style={'margin-top':'6px'}
                    )
                ]
           )
        ]
    )

def filter_data(df, range_select, loc_select, type_select, class_select, n_clicks, keywords):
        
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            df = df[df['full_text'].str.contains(keyword, case=False)]
    df = df[df['localization'].isin(loc_select)]
    df = df[df['type'].isin(type_select)]
    df = df[df[class_select]==1]
    df = df[df['date'] >= range_select[0]]
    filtered_df = df[df['date'] <= range_select[1]]
    
    return filtered_df

def generate_barchart(df, range_select, loc_select, type_select, class_select, n_clicks, keywords):
    
    graph_layout = copy.deepcopy(layout)
    filtered_df = filter_data(df, [init_start_date,init_end_date], loc_select, type_select, class_select, n_clicks, keywords)
    g = filtered_df[['date']]
    g.index = g['date']
    g = g.resample('D').count()
    g.rename(columns={'date':'count'},inplace=True)
    cols = []
    for i in g.index:
        if i.strftime('%Y-%m-%d') >= range_select[0] and i.strftime('%Y-%m-%d') <= range_select[1]:
            cols.append("rgb(214,237,255)")
        else:
            cols.append("rgba(214,237,255,0.3)")
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
            marker=dict(color=cols),
            hovertemplate ='<b>%{x} </b><br>Count: %{y}<extra></extra>'
        ),
    ]
    graph_layout["dragmode"] = 'select'
    graph_layout["selectdirection"] = 'h'
    graph_layout["showlegend"] = False
    graph_layout["height"] = 100
    
    return dict(data=data, layout=graph_layout)

def generate_scatter_map(geo_df, style_select, loc_select, graph_layout):
    
    scatter_layout = copy.deepcopy(layout)
    traces = []
    i = 0
    for filter_type, dff in geo_df.groupby('localization'):
        tweet = dff["full_text"] 
        user_name = dff['user_name']
        user_location = dff['user_location']
        created_at = dff['created_at']
        source = dff['source']
        localization = dff["localization"]
        retweet_count = dff['retweet_count']
        hashtags = dff['hashtags']
        trace = dict(
            type="scattermapbox",
            lat=dff["lat"],
            lon=dff["lon"],
            name=loc_select[i],
            selected=dict(marker={"opacity":1.0,"color":"#d6edff" if style_select=='dark' or style_select=='satellite' else "#171b26"}),
            unselected=dict(marker={"opacity":0.3}),
            hoverinfo="text",
            text='<b>'+tweet+'</b><br>User name: '+user_name+'<br>User location: '+user_location+'<br>Created at: '+created_at.map(str)+
                 '<br>Source: '+source+'<br>Retweet count: '+retweet_count.map(str), # '<br>Localization: '+localization+
            marker=dict(size=4.5,opacity=0.9,color=colors[i]),
            customdata=hashtags
        )
        traces.append(trace)
        i += 1
        
    if graph_layout is not None and "mapbox.center" in graph_layout.keys():
        scatter_layout["mapbox"]["center"]["lon"] = float(graph_layout["mapbox.center"]["lon"])
        scatter_layout["mapbox"]["center"]["lat"] = float(graph_layout["mapbox.center"]["lat"])
        scatter_layout["mapbox"]["zoom"] = float(graph_layout["mapbox.zoom"])
    scatter_layout["mapbox"]["style"] = style_select
    
    return dict(data=traces, layout=scatter_layout)
        
def generate_density_map(geo_df, style_select, graph_layout):        
    
    density_layout = copy.deepcopy(layout)
    trace = [dict(
        type="densitymapbox",
        lat=geo_df["lat"],
        lon=geo_df["lon"],
        #z=geo_df['retweet_count'],
        radius=4,
        opacity=0.8,
        hoverinfo="text",
        hovertext=geo_df["full_text"],
        showscale=False,
    )]
    
    if graph_layout is not None and "mapbox.center" in graph_layout.keys():
        density_layout["mapbox"]["center"]["lon"] = float(graph_layout["mapbox.center"]["lon"])
        density_layout["mapbox"]["center"]["lat"] = float(graph_layout["mapbox.center"]["lat"])
        density_layout["mapbox"]["zoom"] = float(graph_layout["mapbox.zoom"])
    density_layout["mapbox"]["style"] = style_select
    
    return dict(data=trace, layout=density_layout)
    
def generate_hexabin_map(geo_df, style_select, graph_layout):
    
    hexa_layout = copy.deepcopy(layout)
    trace = ff.create_hexbin_mapbox(geo_df, 
                                  lat="lat", 
                                  lon="lon",
                                  nx_hexagon=200, #int(max(30,len(geo_df)/150)), 
                                  min_count=5, 
                                  opacity=0.9, 
                                  labels={"color": "Count"},
    )
    
    if graph_layout is not None and "mapbox.center" in graph_layout.keys():
        hexa_layout["mapbox"]["center"]["lon"] = float(graph_layout["mapbox.center"]["lon"])
        hexa_layout["mapbox"]["center"]["lat"] = float(graph_layout["mapbox.center"]["lat"])
        hexa_layout["mapbox"]["zoom"] = float(graph_layout["mapbox.zoom"])
    hexa_layout["mapbox"]["style"] = style_select
    hexa_layout["clickmode"] = None
    
    return dict(data=trace.data, layout=hexa_layout)

def generate_treemap(filtered_df, geo_select):
    
    if geo_select is None:
        hashtag_list = filtered_df['hashtags'].tolist()
    else:
        hashtag_list = [point["customdata"] for point in geo_select["points"]]
    k = 20
    hashtags = [hashtag for sublist in hashtag_list for hashtag in sublist]
    freq_df = pd.DataFrame(list(FreqDist(hashtags).items()),columns=["hashtag","count"]) 
    freq_df = freq_df.sort_values('count',ascending=False)
    
    fig = go.Figure(go.Treemap(
                        labels=freq_df['hashtag'][:k].tolist(),
                        values=freq_df['count'][:k].tolist(),
                        parents=['']*k,
                        marker_colorscale=px.colors.sequential.Teal,
                        hovertemplate='<b>%{label} </b> <br>Count: %{value}<extra></extra>'))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=10), 
                      height=240,
                      plot_bgcolor="#171b26",
                      paper_bgcolor="#171b26") 
    return fig

def generate_table(filtered_df, geo_select):
    
    if geo_select is None:
        text_df = filtered_df[['full_text']]
    else:
        full_text = [point["text"] for point in geo_select["points"]]
        text_df = pd.DataFrame(full_text,columns=['full_text'])
    text_df.rename(columns={'full_text':'Tweets'},inplace=True)
    
    t = [generate_tweet_div(i) for i in text_df.to_dict('records')]
    return t
    
def generate_tweet_div(tweet):
    return html.P(
        children=[dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(tweet['Tweets']))],
        style={'width':'100%',"background-color":"#242a3b","color":"#7b7d8d",'margin-bottom':'5px'}
    )

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
            build_control_panel(),
        ], 
            xs=12, sm=12, md=1.8, lg=1.8, xl=1.8
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
                            marks=get_marks(df['date'].min(), df['date'].max()),
                            updatemode='mouseup',
                        ), 
                    ]),
                    html.Div([
                        dcc.Loading(children=dcc.Graph(
                            id="barchart",
                        )),
                    ]),
                    html.Div(dcc.Loading(html.Div(id='counter',style={'color':'#7b7d8d','margin-top':'1px'}))),
                ], 
                    xs=12, sm=12, md=9, lg=10, xl=10
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
                                            "data": [], "layout": dict(plot_bgcolor="#171b26",paper_bgcolor="#171b26"),
                                        },
                                    )),
                                ],
                            ),
                            html.Div(
                                id="tweets-outer",
                                children=[
                                    dcc.Loading(children=html.Div(id="tweets-table")),
                                ],
                                style={'overflow-y':'scroll','overflow-x':'none','height':350}
                            ),
                        ],
                    ),
                ], 
                    xs=12, sm=12, md=3, lg=3, xl=3
                ),
            ])
        ], 
            xs=12, sm=12, md=9, lg=9, xl=9
        )
    ], no_gutters=False, justify='start')
], fluid=True)

# Update slider
@app.callback(
    Output("range-slider", "value"),
    Input("barchart", "selectedData")
)
def update_slider(bar_select):
    
    if bar_select is not None:
        nums = [int(point["pointNumber"]) for point in bar_select["points"]]
        start = unix_time(datetime.strptime(init_start_date,'%Y-%m-%d') + timedelta(days=min(nums))) 
        end = unix_time(datetime.strptime(init_start_date,'%Y-%m-%d') + timedelta(days=max(nums)))
        return [start, end]
    else:
        return [init_start, init_end]

# Update barchart 
@app.callback(
        Output('barchart', 'figure'),
    [
        Input('range-slider', 'value'),
        Input('loc-select', 'value'),
        Input('type-select', 'value'),
        Input('class-select', 'value'),
        Input('search-button', 'n_clicks'),
    ], [
        State('text-search', 'value')
    ]
)  
def update_barchart(range_select, loc_select, type_select, class_select, n_clicks, keywords):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    
    return generate_barchart(df, [start,end], loc_select, type_select, class_select, n_clicks, keywords)

# Update map
@app.callback(
        Output('geo-map', 'figure'),
    [
        Input('range-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('loc-select', 'value'),
        Input('type-select', 'value'),
        Input('class-select', 'value'),
        Input('search-button', 'n_clicks'),
    ], [
        State('text-search', 'value'),
        State('geo-map', 'relayoutData')
    ]
)
def update_map(range_select, graph_select, style_select, loc_select, type_select, class_select, n_clicks, keywords, graph_layout):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    filtered_df = filter_data(df, [start,end], loc_select, type_select, class_select, n_clicks, keywords)
   
    if graph_select == 'Scatter':
        geomap = generate_scatter_map(filtered_df, style_select, loc_select, graph_layout)
    elif graph_select == 'Density':
        geomap = generate_density_map(filtered_df, style_select, graph_layout)
    else:
        geomap = generate_hexabin_map(filtered_df, style_select, graph_layout)

    return geomap

# Update content
@app.callback(
    [
        Output('treemap', 'figure'),
        Output('tweets-table', 'children'),
        Output('counter', 'children')
    ], [
        Input('range-slider', 'value'),
        Input('loc-select', 'value'),
        Input('type-select', 'value'),
        Input('class-select', 'value'),
        Input('geo-map', 'selectedData'),
        Input('search-button', 'n_clicks'),
    ], [
        State('text-search', 'value')
    ]
)
def update_content(range_select, loc_select, type_select, class_select, geo_select, n_clicks, keywords):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    filtered_df = filter_data(df, [start,end], loc_select, type_select, class_select, n_clicks, keywords)
    treemap = generate_treemap(filtered_df, geo_select)
    table = generate_table(filtered_df, geo_select)
    if geo_select is None:
        num = len(filtered_df)
    else:
        num = len(geo_select["points"])
    pct = np.round(num/total_count*100,1)
    counter = f'Tweets in selection: {num} ({pct} %)'
    
    return treemap, table, counter

if __name__ == '__main__':
  app.run_server()
