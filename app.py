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
import calendar
import pathlib
import textwrap

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

# Find number of tweets by date
geo_df['Date'] = pd.to_datetime(geo_df['created_at']).dt.date
count_dates = geo_df.groupby('Date').size().values
time_df = geo_df.drop_duplicates(subset="Date").assign(Count=count_dates).sort_values(by='Date').reset_index(drop=True)

# Set graph options
graph_list = ['Point map','Hexagon map']
style_list = ['Light','Dark','Streets','Outdoors','Satellite'] 
loc_types = {'Geotagged coordinates':1,'Geotagged place':2,'Geoparsed from Tweet':3,'Registered user location':4}
loc_list = list(loc_types.keys())

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
                            html.Label("Select Localization Method"),
                            dcc.Dropdown(
                                id='loc-select',
                                options=[{"label": i, "value": i} for i in loc_list],
                                multi=True,
                                value=loc_list
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
           html.Div(f'Total number of relevant Tweets: {total_count}',style={'color':'#7b7d8d'}),
           html.Div(id='counter',style={'color':'#7b7d8d'})
        ],
    )

def generate_geo_map(geo_data, month_select, graph_select, style_select, loc_select, n_clicks, keywords):
    
    if n_clicks > 0 and keywords.strip():
        keywords = keywords.split(', ')
        for keyword in keywords:
            geo_data = geo_data[geo_data['full_text'].str.contains(keyword, case=False)]
    
    types_selected = [loc_types[loc_select[i]] for i in range(len(loc_select))]
    geo_data = geo_data[geo_data['final_coords_type'].isin(types_selected)]
    
    filtered_data = geo_data[geo_data.created_at_month == month_select]
    
    if len(filtered_data) == 0: # no matches
        empty=pd.DataFrame([0, 0]).T
        empty.columns=['lat','long']
        fig = px.scatter_mapbox(empty, lat="lat", lon="long", color_discrete_sequence=['#cbd2d3'])
    
    elif graph_select == 'Point map':
        fig = px.scatter_mapbox(filtered_data, 
                                lat="lat", 
                                lon="lon",
                                hover_name='full_text',
                                hover_data={'lat':False,'lon':False,'user_name':True,'user_location':True,'created_at':True,'source':True,'retweet_count':True},
                                color_discrete_sequence=['#a5d8e6'] if style_select=='dark' else ['#457582'])
    else: # 'Hexagon map':
        fig = ff.create_hexbin_mapbox(filtered_data, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=25, # int(max(25,len(filtered_data)/10)), 
                                      opacity=0.6, 
                                      labels={"color": "Relevant Tweets"},
                                      min_count=1, 
                                      color_continuous_scale='teal',
                                      show_original_data=True, 
                                      original_data_marker=dict(size=5, opacity=1, color='#a5d8e6' if style_select=='dark' else '#457582'))
        
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            center=go.layout.mapbox.Center(
                lat=filtered_data.lat.mean(), lon=filtered_data.lon.mean()
            ),
            zoom=1,
            style=style_select,
        ),
        font=dict(color='#737a8d')
    )
        
    return fig, filtered_data

def generate_line_chart(time_data):
    fig = px.line(time_data,
                  x='Date',
                  y='Count',
                  color_discrete_sequence=['#a5d8e6'])
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linecolor='#ffffff',
        linewidth=1.5
    )
    fig.update_xaxes(
        showgrid=False,
        rangeslider=dict(
            visible=True,
            bgcolor='#737a8d',
            bordercolor='#737a8d',
            thickness=0.125),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
        ]))
    )
    fig.update_layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        font=dict(color='#737a8d'),
    )
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
            className="four columns",
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
                    children="Spatio-Temporal Development of Relevant Tweets"
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
                        dcc.Slider(
                            id='month-slider',
                            min=geo_df['created_at_month'].min(),
                            max=geo_df['created_at_month'].max(),
                            value=geo_df['created_at_month'].min(),
                            marks={int(month): f'{calendar.month_name[int(month)][:3]} {str(year)[:4]}' for year, month in zip(
                                geo_df['created_at_year'], geo_df['created_at_month'])},
                            step=None,
                            updatemode='drag',
                        )
                    ]
                ),
                html.Br(),
                html.Br(),
                html.Div(
                    id="line-chart-outer",
                    children=[
                        html.P(
                            id="line-chart-title",
                            children="Number of Relevant Tweets"
                        ),
                        dcc.Graph(
                            id="line-chart",
                            figure=generate_line_chart(time_df)
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    [
        Output('geo-map', 'figure'),
        Output('counter', 'children')
    ], [
        Input('month-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
        Input('loc-select', 'value'),
        Input('search-button', 'n_clicks'),
        State('text-search', 'value')
    ],
)
def update_geo_map(month_select, graph_select, style_select, loc_select, n_clicks, keywords):
    
    figure, filtered_data = generate_geo_map(geo_df, month_select, graph_select, style_select.lower(), loc_select, n_clicks, keywords)
    
    return figure, f'Number of relevant Tweets in selection: {len(filtered_data)}'

if __name__ == '__main__':
    app.run_server()
