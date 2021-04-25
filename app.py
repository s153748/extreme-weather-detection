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
df = pd.read_csv('data/final_labelled_tweets.csv')

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

def build_upper_left_panel():
    return html.Div(
        id="upper-left",
        className="six columns", 
        children=[
            html.P(
                className="section-title",
                children="Choose graph type to inspect the Tweets in different ways",
            ),
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
                    ),
                ],
            ),
        ],
    )

def generate_geo_map(geo_data, month_select, graph_select):
    
    filtered_data = geo_data[geo_data.created_at_month == month_select]
    
    if graph_select == 'Point map':
        fig = px.scatter_mapbox(filtered_data, 
                                lat="lat", 
                                lon="lon",
                                hover_name='full_text',
                                hover_data=['user_location','created_at','retweet_count'],
                                color_discrete_sequence=['#dae1f2'])
    elif graph_select == 'Hexagon map':
        fig = ff.create_hexbin_mapbox(data_frame=filtered_data, 
                                      lat="lat", 
                                      lon="lon",
                                      nx_hexagon=int(max(3,len(filtered_data)/10)), 
                                      opacity=0.6, 
                                      labels={"color": "Relevant Tweets"},
                                      min_count=1, 
                                      color_continuous_scale='teal',
                                      show_original_data=True, 
                                      original_data_marker=dict(size=5, opacity=1, color="#dae1f2")
        )
    else:
        fig = px.scatter_mapbox()
        
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=filtered_data.lat.mean(), lon=filtered_data.lon.mean()
            ),
            pitch=5,
            zoom=2,
            style="mapbox://styles/plotlymapbox/cjvppq1jl1ips1co3j12b9hex",
        ),
        font=dict(color='#737a8d')
    )
        
    return fig

def generate_line_chart(time_data):
    fig = px.line(time_data,
                  x='Date',
                  y='Count',
                  hover_data=['full_text'],
                  color_discrete_sequence=['#dae1f2'])
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=True,
        rangeselector=dict(buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
    fig.update_layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        font=dict(color='#737a8d')
    )
    return fig

# Set up the layout
app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Extreme Weather Event Detection"),
                html.A([
                    html.Img(
                        src=app.get_asset_url("GitHub-Mark-Light-64px.png")
                    ), href=githublink)
                ])
                #html.A(id='gh-link',
                #       children=['Github'], 
                #       href=githublink, 
                #       style={'color': 'white', 'border': 'solid 1px white', 'font-size':'12px'}
                #)  
            ]
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                build_upper_left_panel(),
                html.Div(
                    id="geo-map-outer",
                    className="six columns",
                    children=[
                        html.P(
                            id="map-title",
                            children="Spatio-Temporal Development of Relevant Tweets"
                        ),
                        html.Div(
                            id="geo-map-loading-outer",
                            children=[
                                dcc.Loading(
                                    id="loading",
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
                                            step=None
                                        )
                                    ]
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
       html.Div(
            id="lower-container",
            className="row",
            children=[
                html.Div(
                    id="line-chart-outer",
                    className="six columns",
                    children=[
                        html.P(
                            id="line-chart-title",
                            children="Number of Relevant Tweets"
                        ),
                        html.Div(
                            id="line-chart-loading-outer",
                            children=[
                                dcc.Loading(
                                    id="loading-line-chart",
                                    children=[
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
            ]
        )
    ]
)

@app.callback(
    Output('geo-map', 'figure'),
    [
        Input('month-slider', 'value'),
        Input("graph-select", "value"),
    ],
)
def update_geo_map(month_select, graph_select):
    
    return generate_geo_map(geo_df, month_select, graph_select)

if __name__ == '__main__':
    app.run_server()
