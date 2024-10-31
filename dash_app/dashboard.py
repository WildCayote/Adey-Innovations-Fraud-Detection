from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import requests
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Set a new color palette for a fresh look
color_palette = {
    'background': '#F5F8FA',
    'text': '#34495E',
    'box_border': '#2980B9',
    'transaction_bg': '#E8F8F5',  
    'fraud_cases_bg': '#FDEDEC',  
    'fraud_rate_bg': '#F9E79F',   
    'box_text': '#5D6D7E',
    'trend_line': '#16A085',
    'device_chart': '#D35400',
    'browser_chart': '#8E44AD'
}

# Define the layout of the dashboard
app.layout = html.Div(style={'backgroundColor': color_palette['background'], 'padding': '10px'}, children=[
    dbc.Container(fluid=False, children=[
        # Header section
        html.Div(style={'textAlign': 'center', 'padding': '20px'}, children=[
            html.H1(children='Real-Time Fraud Insights', style={'color': color_palette['text']}),
            html.Hr(),
        ]),

        # Summary boxes for key metrics (two rows of boxes)
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='transaction-total', style={'color': color_palette['box_text']}),
                    html.P('All Transactions', style={'fontSize': '14px', 'color': '#7D8A8A'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}',
                'borderRadius': '5px',
                'backgroundColor': color_palette['transaction_bg']
            }), width=6),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='fraud-total', style={'color': color_palette['box_text']}),
                    html.P('Total Fraudulent Activities', style={'fontSize': '14px', 'color': '#7D8A8A'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}',
                'borderRadius': '5px',
                'backgroundColor': color_palette['fraud_cases_bg']
            }), width=6),
        ], className="mb-2"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='fraud-rate', style={'color': color_palette['box_text']}),
                    html.P('Fraud Rate', style={'fontSize': '14px', 'color': '#7D8A8A'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}',
                'borderRadius': '5px',
                'backgroundColor': color_palette['fraud_rate_bg']
            }), width=12),
        ], className="mb-4"),

        # Trend analysis chart (stacked with map)
        dbc.Row([
            dbc.Col(dcc.Graph(id='trend-analysis-chart'), width=6),
            dbc.Col(dcc.Graph(id='fraud-locations-map', style={'height': '400px'}), width=6),
        ], className="mb-4"),

        # Device and browser charts (2x2 grid layout)
        dbc.Row([
            dbc.Col(dcc.Graph(id='top-devices-chart'), width=6),
            dbc.Col(dcc.Graph(id='top-browsers-chart'), width=6),
        ])
    ])
])

# Callbacks for updating summary data
@app.callback(
    [Output('transaction-total', 'children'),
     Output('fraud-total', 'children'),
     Output('fraud-rate', 'children')],
    Input('transaction-total', 'id')  # Dummy input to trigger callback
)
def update_summary(_):
    """Fetch and update key metrics from the API."""
    try:
        response = requests.get('http://127.0.0.1:5000/fraud_summary_statistics')
        response.raise_for_status()
        summary_data = response.json()
        return (
            f'Total: {summary_data["total_transactions"]}',
            f'Fraudulent Cases: {summary_data["total_fraud_cases"]}',
            f'Rate: {summary_data["fraud_percentage"]:.2f}%'
        )
    except Exception as e:
        return ('Error',) * 3  # Return error message for all outputs

# Callback for fraud trends line chart
@app.callback(
    Output('trend-analysis-chart', 'figure'),
    Input('trend-analysis-chart', 'id')  # Dummy input to trigger callback
)
def update_fraud_trends(_):
    """Fetch and update trend line chart for fraudulent activities over time."""
    try:
        response = requests.get('http://127.0.0.1:5000/fraud_trends_over_time')
        response.raise_for_status()
        trend_data = pd.DataFrame(response.json())
        fig = px.line(
            trend_data,
            x='purchase_date',
            y='fraud_cases',
            title='Monthly Fraud Cases Trend',
            line_shape='spline',
            markers=True,
            template='plotly_white',
            color_discrete_sequence=[color_palette['trend_line']]
        )
        fig.update_traces(hovertemplate='Date: %{x}<br>Fraud Cases: %{y}')
        return fig
    except Exception as e:
        return px.line(title="Error loading data.")  # Return an empty line chart on error

# Callback for fraud cases by location map
@app.callback(
    Output('fraud-locations-map', 'figure'),
    Input('fraud-locations-map', 'id')  # Dummy input to trigger callback
)
def update_fraud_by_location(_):
    """Fetch and update map visualization for fraud distribution by country."""
    try:
        response = requests.get('http://127.0.0.1:5000/fraud_statistics_by_location')
        response.raise_for_status()
        location_data = pd.DataFrame(response.json())
        fig = px.choropleth(
            location_data,
            locations='country',
            locationmode='country names',
            color='fraud_cases',
            title='Fraud Distribution by Country',
            color_continuous_scale='Reds',
            template='plotly_white',
            labels={'fraud_cases': 'Fraud Cases'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        return fig
    except Exception as e:
        return px.choropleth(title="Error loading data.")  # Return an empty map on error

# Callback for top devices and browsers charts
@app.callback(
    [Output('top-devices-chart', 'figure'),
     Output('top-browsers-chart', 'figure')],
    Input('top-devices-chart', 'id')  # Dummy input to trigger callback
)
def update_fraud_by_device_browser(_):
    """Fetch and update charts for top devices and browsers involved in fraud."""
    try:
        response = requests.get('http://127.0.0.1:5000/top_fraud_device_browser_combinations')
        response.raise_for_status()
        device_browser_data = pd.DataFrame(response.json())

        # Top 10 devices and browsers
        top_devices = device_browser_data.nlargest(10, 'fraud_cases')
        top_browsers = device_browser_data.groupby('browser').agg({'fraud_cases': 'sum'}).reset_index()
        top_browsers = top_browsers.nlargest(10, 'fraud_cases')

        # Device bar chart
        device_fig = px.bar(
            top_devices,
            x='device_id',
            y='fraud_cases',
            title='Top Fraud Devices',
            labels={'device_id': 'Device', 'fraud_cases': 'Cases'},
            template='plotly_white',
            color_discrete_sequence=[color_palette['device_chart']]
        )
        device_fig.update_traces(hovertemplate='Device: %{x}<br>Cases: %{y}')

        # Browser bar chart
        browser_fig = px.bar(
            top_browsers,
            x='browser',
            y='fraud_cases',
            title='Top Browsers for Fraud',
            labels={'browser': 'Browser', 'fraud_cases': 'Cases'},
            template='plotly_white',
            color_discrete_sequence=[color_palette['browser_chart']]
        )
        browser_fig.update_traces(hovertemplate='Browser: %{x}<br>Cases: %{y}')

        return device_fig, browser_fig
    except Exception as e:
        return (px.bar(title="Error loading device data."), px.bar(title="Error loading browser data."))  # Return empty charts on error

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
