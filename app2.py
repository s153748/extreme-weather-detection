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
df['type'] = ""
for i in range(len(df)):
    if df.retweeted[i]:
        df['type'][i] = 'Retweet'
    else:
        df['type'][i] = 'Tweet' 

def unix_time(dt):
    return (dt-datetime.utcfromtimestamp(0)).total_seconds() 
init_start = unix_time(df['date'].min())
init_end = unix_time(df['date'].max())
init_start_date = datetime.utcfromtimestamp(init_start).strftime('%Y-%m-%d')
init_end_date = datetime.utcfromtimestamp(init_end).strftime('%Y-%m-%d')

def get_marks(start, end):
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return {int(unix_time(m)): (str(m.strftime('%b %Y'))) for m in result}

# Set graph options
graph_list = ['Scatter map','Hexagon map']
style_list = ['light','dark','streets','outdoors','satellite'] 
filter_options = {'localization':['Geotagged coordinates','Geotagged place','Geoparsed from Tweet','Registered user location'],
                  'type':['Tweet','Retweet']}
colors = ['#ffa600','#ef5675','#8073ac','#35978f']

# Create global chart template
layout = dict(
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#171b26",
    paper_bgcolor="#171b26",
    hovermode="closest",
    dragmode='select',
    mapbox=dict(accesstoken=mapbox_access_token,
                style='light',
                center=go.layout.mapbox.Center(lat=20, lon=-3),
                zoom=0.7,
    ),
    legend=dict(bgcolor="#cbd2d3",
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
                                options=[{"label": i.capitalize(), "value": i} for i in style_list],
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
                        id="filter-select-outer",
                        children=[
                            html.Label("Filter by"),
                            #dcc.Dropdown(
                            dcc.RadioItems(
                                id="filter-select",
                                options=[{'label': k.capitalize(), 'value': k} for k in filter_options.keys()],
                                value='localization',
                                labelStyle={"display": "block"},
                            ),
                            dcc.Dropdown(id='option-select',
                                         multi=True),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="control-row-4",
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
                            html.Div('Tweets in selection:',style={'color':'#7b7d8d','fontsize':'9px','margin-top':'20px'}),
                            html.Div(id='counter',style={'color':'#7b7d8d','fontsize':'9px','margin-top':'1px'}),
                        ]
                    )
                ]
           )
        ]
    )

def filter_data(df, range_select, filter_select, option_select, n_clicks, keywords):
        
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            df = df[df['full_text'].str.contains(keyword, case=False)]
    df = df[df[filter_select].isin(option_select)]
    df = df[df['date'] >= range_select[0]]
    filtered_df = df[df['date'] <= range_select[1]]
    
    return filtered_df

def generate_barchart(df, range_select, filter_select, option_select, n_clicks, keywords):
    
    graph_layout = copy.deepcopy(layout)
    filtered_df = filter_data(df, [init_start_date,init_end_date], filter_select, option_select, n_clicks, keywords)
    g = filtered_df[['date']]
    g.index = g['date']
    g = g.resample('D').count()
    g.rename(columns={'date':'count'},inplace=True)
    cols = []
    for i in g.index:
        if i.strftime('%Y-%m-%d') >= range_select[0] and i.strftime('%Y-%m-%d') <= range_select[1]:
            cols.append("rgb(197,221,240)")
        else:
            cols.append("rgba(197,221,240,0.2)")
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
    graph_layout["selectdirection"] = 'h'
    graph_layout["showlegend"] = False
    graph_layout["height"] = 100
    
    fig = dict(data=data, layout=graph_layout)
    
    return fig

def generate_geo_map(geo_df, graph_select, style_select, filter_select, graph_layout):
    
    if graph_select == 'Scatter map':
        traces = []
        i = 0
        for filter_type, dff in geo_df.groupby(filter_select):
            tweet = dff["full_text"]
            hashtags = dff['hashtags']
            localization = dff["localization"]
            user_name = dff['user_name']
            user_location = dff['user_location']
            created_at = dff['created_at']
            source = dff['source']
            tweettype = dff['type']
            #retweet_count = dff['retweet_count']
            trace = dict(
                type="scattermapbox",
                lat=dff["lat"],
                lon=dff["lon"],
                name=filter_options[filter_select][i],
                selected=dict(marker={"color":"#c5ddf0" if style_select=='dark' else "#171b26"}),
                hoverinfo="text",
                text='<b>'+tweet+'</b><br>User name: '+user_name+'<br>User location: '+user_location+
                     '<br>Created at: '+created_at+'<br>Source: '+source+'<br>Localization: '+localization+
                     '<br>Type: '+tweettype,#'<br>retweet_count: '+retweet_count+'<br>',
                marker=dict(size=4,opacity=0.7,color=colors[i]),
                customdata=hashtags,
            )
            traces.append(trace)
            i += 1
        
        if graph_layout is not None:
            if "mapbox.center" in graph_layout.keys():
                layout["mapbox"]["center"]["lon"] = float(graph_layout["mapbox.center"]["lon"])
                layout["mapbox"]["center"]["lat"] = float(graph_layout["mapbox.center"]["lat"])
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=20), 
                      height=240,
                      width=240,
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
    
    table = dash_table.DataTable( 
        #id="tweets-table",
        columns=[{"name": i, "id": i} for i in text_df.columns],
        data=text_df.to_dict('records'),
        page_size=5,
        style_cell={'whiteSpace':'normal','height':'auto','width':'240px',"background-color":"#242a3b","color":"#7b7d8d"},
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
                            marks=get_marks(df['date'].min(), df['date'].max()),
                            updatemode='mouseup',
                        ), 
                    ]),
                    #html.Div(id='output-range-slider',style={'color':'#7b7d8d','fontsize':'8px','margin-bottom':'1px'}),
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


# Set filter options and value
@app.callback(
    [
        Output("option-select", "options"),
        Output("option-select", "value"),
    ], [
        Input("filter-select", "value")   
    ]
)
def set_options(filter_select):
    options = [{"label": i, "value": i} for i in filter_options[filter_select]]
    value = filter_options[filter_select]
    return options, value
    
# Update slider based on barchart
@app.callback(
    Output("range-slider", "value"),
    Input("barchart", "selectedData")
)
def update_slider(bar_select):
    
    if bar_select is not None:
        nums = [int(point["pointNumber"]) for point in bar_select["points"]]
        start = unix_time(datetime.strptime('2013-01-27','%Y-%m-%d') + timedelta(days=min(nums)))
        end = unix_time(datetime.strptime('2013-01-28','%Y-%m-%d') + timedelta(days=max(nums)))
        return [start, end]
    else:
        return [init_start, init_end]

# Update barchart and geomap based on configurations
@app.callback(
    [
        Output('barchart', 'figure'),
        Output('geo-map', 'figure'),
        Output('counter', 'children'),
        #Output('output-range-slider', 'children'),
    ], [
        Input('range-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('filter-select', 'value'),
        Input('option-select', 'value'),
        Input('search-button', 'n_clicks'),
    ], [
        State('text-search', 'value'),
        State('geo-map', 'relayoutData')
    ], 
)
def update_visuals(range_select, graph_select, style_select, filter_select, option_select, n_clicks, keywords, graph_layout):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    filtered_df = filter_data(df, [start,end], filter_select, option_select, n_clicks, keywords)
    barchart = generate_barchart(df, [start,end], filter_select, option_select, n_clicks, keywords)
    geomap = generate_geo_map(filtered_df, graph_select, style_select, filter_select, graph_layout)
    pct = np.round(len(filtered_df)/total_count*100,1)
    counter = f'{len(filtered_df)} ({pct}%)'
    #period = f'Selected period: {pd.to_datetime(start).strftime("%b %d, %Y")} - {pd.to_datetime(end).strftime("%b %d, %Y")}'

    return barchart, geomap, counter #period    

# Update table and treemap based on configurations and map
@app.callback(
    [
        Output('treemap', 'figure'),
        Output('tweets-table', 'children')
    ], [
        Input('geo-map', 'selectedData'),
        Input('range-slider', 'value'),
        Input('filter-select', 'value'),
        Input('option-select', 'value'),
        Input('search-button', 'n_clicks'),
    ], [
        State('text-search', 'value'),
    ], 
)
def update_content(geo_select, range_select, filter_select, option_select, n_clicks, keywords):
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    filtered_df = filter_data(df, [start,end], filter_select, option_select, n_clicks, keywords)
    treemap = generate_treemap(filtered_df, geo_select)
    table = generate_table(filtered_df, geo_select)
    
    return treemap, table

if __name__ == '__main__':
    app.run_server()
