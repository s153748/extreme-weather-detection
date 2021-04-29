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
import textwrap
import calendar
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Initiate app
app = dash.Dash(
    __name__,
    meta_tags=[{
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
    }],
)
server = app.server
app.config.suppress_callback_exceptions = True

githublink = 'https://github.com/s153748/extreme-weather-detection'
mapbox_access_token = open(".mapbox_token.txt").read()

# Load data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data") 
df = pd.read_csv(DATA_PATH.joinpath("final_coords_tweets.csv")) 

wrapper = textwrap.TextWrapper(width=50)

# Data prep
for i in range(len(df)):
    try:
        df['final_coords'][i] = eval(df['final_coords'][i])
        df['full_text'][i] = "<br>".join(wrapper.wrap(text=df['full_text'][i]))
    except:
        df['final_coords'][i] = np.nan
geo_df = df[~df['final_coords'].isna()].reset_index(drop=True)
geo_df = geo_df[geo_df['relevant'] == 1].reset_index(drop=True)
total_count = len(geo_df)

# Get coordinates
geo_df['lat'] = [geo_df['final_coords'][i][0] for i in range(len(geo_df))]
geo_df['lon'] = [geo_df['final_coords'][i][1] for i in range(len(geo_df))]

# Group by date
geo_df['Date'] = pd.to_datetime(geo_df['created_at']).dt.date

# Date slider prep
geo_df['Date'] = pd.to_datetime(geo_df['Date'])
def unix_time(dt):
    return (dt-datetime.utcfromtimestamp(0)).total_seconds() 

def get_marks(start, end):
    result = []
    current = start
    while current <= end:
        result.append(current)
        current += relativedelta(months=1)
    return {int(unix_time(m)): (str(m.strftime('%Y-%m'))) for m in result}

# Set graph options
graph_list = ['Scatter map','Hexagon map']
style_list = ['Light','Dark','Streets','Outdoors','Satellite'] 
loc_types = {'Geotagged coordinates':1,'Geotagged place':2,'Geoparsed from Tweet':3,'Registered user location':4} # localization methods
loc_list = list(loc_types.keys())
cmap = {1:'#253494',2:'#2c7fb8',3:'#41b6c4',4:'#c7e9b4'} # localization method colors

def build_control_panel():
    return html.Div(
        id="control-panel",
        children=[
            html.P(
                className="section-title",
                children="Configurations",
            ),
            html.Br(),
            html.Div(
                className="control-row-1",
                children=[
                    html.Div(
                        id="graph-select-outer",
                        children=[
                            html.Label("Select Graph Type"),
                            dcc.Dropdown(
                                id="graph-select",
                                options=[{"label": i, "value": i} for i in graph_list],
                                value=graph_list[0],
                            ),
                        ],
                    )
                ]
            ),
            html.Br(),
            html.Div(
                className="control-row-2",
                children=[
                    html.Div(
                        id="style-select-outer",
                        children=[
                            html.Label("Select Map Style"),
                            dcc.Dropdown(
                                id="style-select",
                                options=[{"label": i, "value": i} for i in style_list],
                                value=style_list[0],
                            ),
                        ],
                    ),
                ],
            ),
            html.Br(),
            html.Div(
                className="control-row-3",
                children=[
                    html.Div(
                        id="loc-select-outer",
                        children=[
                            html.Label("Select Localization Methods"),
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
            html.Br(),
            html.Div(
                className="control-row-4",
                children=[
                    html.Div(
                        id="text-search-outer",
                        children=[
                            html.Label("Filter on Keywords"),
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

def generate_geo_map(geo_data, range_select, graph_select, style_select, loc_select, n_clicks, keywords):
    
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            geo_data = geo_data[geo_data['full_text'].str.contains(keyword, case=False)]
    
    selected = [loc_types[loc_select[i]] for i in range(len(loc_select))] 
    geo_data = geo_data[geo_data['final_coords_type'].isin(selected)]
    colors = [cmap[selected[i]] for i in range(len(selected))]
    
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    geo_data = geo_data[geo_data['Date'] >= start]
    filtered_data = geo_data[geo_data['Date'] <= end]    
    
    if len(filtered_data) == 0: # no matches
        empty=pd.DataFrame([0, 0]).T
        empty.columns=['lat','long']
        fig = px.scatter_mapbox(empty, lat="lat", lon="long", color_discrete_sequence=['#cbd2d3'], opacity=0)
    
    elif graph_select == 'Scatter map':
        fig = px.scatter_mapbox(filtered_data, 
                                lat="lat", 
                                lon="lon",
                                color='final_coords_type', # localization methods
                                hover_name='full_text',
                                hover_data={'lat':False,'lon':False,'user_name':True,'user_location':True,'created_at':True,'source':True,'retweet_count':True},
                                #color_discrete_sequence=['#a5d8e6'] if style_select=='dark' else ['#457582'],
                                color_continuous_scale=colors,
                               )
        fig.update(layout_coloraxis_showscale=False)
    elif graph_select == 'Hexagon map':
        fig = ff.create_hexbin_mapbox(filtered_data, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=75, # int(max(25,len(filtered_data)/10)), 
                                      opacity=0.6, 
                                      labels={"color": "Tweets"},
                                      min_count=1, 
                                      color_continuous_scale='teal',
                                      show_original_data=True, 
                                      original_data_marker=dict(size=5, opacity=0.6, color='#a5d8e6' if style_select=='dark' else '#457582')
                                     )
    fig.update_layout(
        margin=dict(l=0, r=0, t=27, b=0), 
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            center=go.layout.mapbox.Center(lat=0, lon=0),
            zoom=1,
            style=style_select,
        ),
        font=dict(color='#737a8d')
    )
        
    return fig, filtered_data

def generate_line_chart(filtered_data):
    
    count_dates = filtered_data.groupby('Date').size().values
    time_data = filtered_data.drop_duplicates(subset="Date").assign(Count=count_dates).sort_values(by='Date').reset_index(drop=True)
    
    fig = px.line(time_data,
                  x='Date',
                  y='Count',
                  color_discrete_sequence=['#cbd2d3'],
                  height=200)
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linecolor='#ffffff',
        linewidth=1.5)
    fig.update_xaxes(
        showgrid=False)
    fig.update_layout(
        font=dict(color='#737a8d'))
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
            id="right-column",
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
                                "data": [],
                                "layout": dict(
                                    plot_bgcolor="#171b26",
                                    paper_bgcolor="#171b26",
                                ),
                            },
                        ),
                        dcc.RangeSlider(
                            id='range-slider',
                            min=unix_time(geo_df['Date'].min()),
                            max=unix_time(geo_df['Date'].max()),
                            value=[unix_time(geo_df['Date'].min()), unix_time(geo_df['Date'].max())],
                            marks=get_marks(geo_df['Date'].min(),geo_df['Date'].max()),
                            updatemode='drag',
                            dots=False,
                        ),
                        dcc.Graph(
                            id="line-chart",
                            figure={
                                "data": [],
                                "layout": dict(
                                    plot_bgcolor="#171b26",
                                    paper_bgcolor="#171b26",
                                ),
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
        Output('line-chart', 'figure'),
        Output('output-range-slider', 'children'),
        Output('counter', 'children')
    ], [
        Input('range-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('loc-select', 'value'),
        Input('search-button', 'n_clicks'),
        State('text-search', 'value')
    ],
)
def update_geo_map(range_select, graph_select, style_select, loc_select, n_clicks, keywords):
    
    geo_map, filtered_data = generate_geo_map(geo_df, range_select, graph_select, style_select.lower(), loc_select, n_clicks, keywords)
    line_chart = generate_line_chart(filtered_data)
    start = datetime.utcfromtimestamp(range_select[0]).strftime('%Y-%m-%d')
    end = datetime.utcfromtimestamp(range_select[1]).strftime('%Y-%m-%d')
    
    return geo_map, line_chart, f'Period: {start} - {end}', f'Tweets in selection: {len(filtered_data)}', 

if __name__ == '__main__':
    app.run_server()
