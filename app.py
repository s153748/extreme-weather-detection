import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# Define variables
tabtitle='events'
myheading='Extreme Weather Detection'
githublink='https://github.com/s153748/extreme-weather-detection'

# Set up visualization

# Initiate app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = tabtitle

# Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading),
    html.A('Code on Github', href=githublink),
    html.Br(),
    ]
)

if __name__ == '__main__':
    app.run_server()
