# Import required packages
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

# Add Dataframe
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "NYC", "MTL", "NYC"]
})
# Add a bar graph figure
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# dash object
app = dash.Dash()

# layout
app.layout = html.Div(children=[
    html.H1(
        children='Dashboard',
        style={
            'textAlign': 'center'
        }
    ),

    # Create dropdown
    dcc.Dropdown(options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC' # Providing a vallue to dropdown
    ),   
    
    # Bar graph
    dcc.Graph(id='example-graph-2',figure=fig)

]1


# Run Application
if __name__ == '__main__':
    app.run_server()