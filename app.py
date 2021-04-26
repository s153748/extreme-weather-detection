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
mapbox_access_token = 'pk.eyJ1IjoiczE1Mzc0OCIsImEiOiJja25wcDlwdjYxcWJmMnFueDhhbHdreTlmIn0.DXfj5S2H91AZEPG1JnHbxg'

# Load data
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data") 
df = pd.read_csv(DATA_PATH.joinpath("final_labelled_tweets.csv")) 

# Data prep
df.dropna(subset=['tokens'], inplace=True)
df['tokens'] = [literal_eval(s) for s in df['tokens']]
for i in range(len(df)):
    try:
        df['geo'][i] = eval(df['geo'][i])
    except:
        df['geo'][i] = np.nan
    try:
        df['place'][i] = eval(df['place'][i])
    except:
        df['place'][i] = np.nan
    
geo_df = df[~df['geo'].isna()].reset_index(drop=True)
geo_df = geo_df[geo_df['relevant'] == 1].reset_index(drop=True)
for i in range(len(geo_df)):
    try:
        geo_df['geo'][i] = eval(geo_df['geo'][i])
    except:
        geo_df['geo'][i] = geo_df['geo'][i]

# Get coordinates
geo_df['lat'] = [geo_df['geo'][i]['coordinates'][0] for i in range(len(geo_df))]
geo_df['lon'] = [geo_df['geo'][i]['coordinates'][1] for i in range(len(geo_df))]

# Find number of tweets by date
geo_df['Date'] = pd.to_datetime(geo_df['created_at']).dt.date
count_dates = geo_df.groupby('Date').size().values
time_df = geo_df.drop_duplicates(subset="Date").assign(Count=count_dates).sort_values(by='Date').reset_index(drop=True)

# Set graph options
graph_list = ['Point map','Hexagon map']
style_list = ["carto-darkmatter",'carto-positron','open-street-map']

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
                        id="graph-style-outer",
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
        ],
    )

def generate_geo_map(geo_data, month_select, graph_select, style_select):
    
    filtered_data = geo_data[geo_data.created_at_month == month_select]
    
    if graph_select == 'Point map':
        fig = px.scatter_mapbox(filtered_data, 
                                lat="lat", 
                                lon="lon",
                                hover_name='full_text',
                                hover_data=['user_location','created_at','retweet_count'],
                                color_discrete_sequence=['#a5d8e6'] if style_select=='carto-darkmatter' else ['#457582'])
    else:
        fig = ff.create_hexbin_mapbox(data_frame=filtered_data, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=int(max(5,len(filtered_data)/15)), 
                                      opacity=0.6, 
                                      labels={"color": "Relevant Tweets"},
                                      min_count=1, 
                                      color_continuous_scale='teal',
                                      show_original_data=True, 
                                      original_data_marker=dict(size=5, opacity=1, color='#a5d8e6' if style_select=='carto-darkmatter' else '#457582'))
        
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
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
            zoom=3,
            style=style_select,
        ),
        font=dict(color='#737a8d')
    )
        
    return fig

def generate_line_chart(time_data):
    fig = px.line(time_data,
                  x='Date',
                  y='Count',
                  hover_data=['full_text'],
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
                #dict(count=6, label="6m", step="month", stepmode="backward"),
                #dict(count=1, label="1y", step="year", stepmode="backward"),
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
                    className="row div-for-charts bg-grey",
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
                            verticalHeight=450
                        )
                    ]
                ),
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
    Output('geo-map-outer', 'children'),
    [
        Input('month-slider', 'value'),
        Input('graph-select', 'value'),
        Input('style-select', 'value'),
    ],
)
def update_geo_map(month_select, graph_select, style_select):
    
    return generate_geo_map(geo_df, month_select, graph_select, style_select)

if __name__ == '__main__':
    app.run_server()
