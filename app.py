import dash
import dash_bootstrap_components as dbc

# Initiate the app
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Set up the layout
app.layout = dbc.Alert(
    "Test", className="m-5"
)

if __name__ == '__main__':
    app.run_server()
