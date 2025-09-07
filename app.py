# Import the necessary libraries
from dash import Dash, html, dcc, Input, Output, dash_table
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from model_utils import get_available_countries, get_country_data,get_global_data, prophet_forecast,get_model_comparison,random_forest_forecast

app = Dash(__name__,suppress_callback_exceptions=True)
server = app.server

country_options =   [{'label':country,'value':country} for country in get_available_countries()]

app.layout = html.Div([
    html.H1(children='COVID-19 Interactive Dashboard',
            style={
                'backgroundColor': '#2c3e50',
                'padding': '20px',
                'textAlign': 'center',
                'color': 'white',
                'marginBottom': '20px'
            }),
    
    html.Div([
        dcc.Dropdown(id='country_dd', options=country_options,
                    value='US' if 'US' in [option['value'] for option in country_options] else country_options[0]['value'],
                    style={
                        'width': '50%', 
                        'margin': 'auto',
                        'backgroundColor': '#ecf0f1',
                        'borderRadius': '5px'
                    })
    ], style={'padding': '20px', 'backgroundColor': '#34495e'}),
    
    dcc.Tabs(id='tabs', value='tab-forecast', children=[
        dcc.Tab(label='📈 Forecast & Trends', value='tab-forecast',
               style={'backgroundColor': '#e74c3c', 'color': 'white'},
               selected_style={'backgroundColor': '#c0392b'}),
        dcc.Tab(label='🌍 World Map', value='tab-map',
               style={'backgroundColor': '#3498db', 'color': 'white'},
               selected_style={'backgroundColor': '#2980b9'}),
        dcc.Tab(label='📊 Data Table', value='tab-table',
               style={'backgroundColor': '#2ecc71', 'color': 'white'},
               selected_style={'backgroundColor': '#27ae60'}),
        dcc.Tab(label='💡 Insights', value='tab-insights',
               style={'backgroundColor': '#f39c12', 'color': 'white'},
               selected_style={'backgroundColor': '#d35400'}),
        dcc.Tab(label='🤖 Model Comparison', value='tab-comparison',
               style={'backgroundColor': '#9b59b6', 'color': 'white'},
               selected_style={'backgroundColor': '#8e44ad'})
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='tabs-content', style={
        'padding': '20px',
        'backgroundColor': '#ecf0f1',
        'borderRadius': '10px',
        'minHeight': '400px'
    }),
])


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),

	Input('country_dd', 'value')]
)

def render_tab_content(tab, selected_country):
	#Selection of tabs
	if tab == 'tab-forecast':
		return html.Div([
			dcc.Graph(id='forecast-plot'),
			dcc.Graph(id='trends-plot')
		])
	elif tab == 'tab-map':
		return html.Div([
			dcc.Graph(id='world-map')
		])
	elif tab == 'tab-table':
		return html.Div([
			dcc.Graph(id='forecast-table')
		])
	elif tab == 'tab-insights':
		return html.Div([
			html.Div(id='model-metrics', style={'padding': '20px'})
		])
	elif tab == 'tab-comparison':  # ← ADD THIS SECTION
		return html.Div([
			dcc.Graph(id='model-comparison-plot'),
			html.Div(id='model-metrics-comparison', style={'padding': '20px'})
		])
# Callback for Forecast Plot
@app.callback(
     Output('forecast-plot', 'figure'),
    [Input('country_dd', 'value'),
    Input('tabs', 'value')]
)
def update_forecast_plot(selected_country,current_tab):
	# ADD THIS: Prevent update if not on the correct tab
	if current_tab != 'tab-forecast':
		raise PreventUpdate
	# Get historical data AND forecast data
	historical_df = get_country_data(selected_country)
	prophet_df, future_dates, forecast_values,_ = prophet_forecast(selected_country)
	# Create  plot with forecast data
	fig = go.Figure()
	# Add historical data
	fig.add_trace(go.Scatter(
		x=historical_df['Date'],
		y=historical_df['Cases'],
		mode='lines',
		name='Historical Data',
		line=dict(color='blue')
	))
	# Add forecasted data
	fig.add_trace(go.Scatter(
		x=future_dates,
		y=forecast_values,
		mode='lines',
		name='Prophet Forecast',
		line=dict(color='red', dash='dash')
	))
	fig.update_layout(title=f'COVID-19 Forecast for {selected_country}',
						xaxis_title='Date', yaxis_title='Cases')
	return fig
# Callback for Trends Plot (7-day average)
@app.callback(
    Output('trends-plot', 'figure'),
    [Input('country_dd', 'value'),
    Input('tabs', 'value')]
)
def update_trends_plot(selected_country,current_tab):
	if current_tab != 'tab-forecast':
		raise PreventUpdate
	historical_df = get_country_data(selected_country)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Cases'],
							mode='lines', name='Actual Cases', line=dict(color='blue')))
	fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['7Day_MA'],
							mode='lines', name='7-Day Average', line=dict(color='green', dash='dot')))
	fig.update_layout(title=f'Trend Analysis for {selected_country}',
						xaxis_title='Date', yaxis_title='Cases')
	return fig

# Callback for World Map
@app.callback(
    Output('world-map', 'figure'),
    Input('tabs', 'value')
)
def update_world_map(current_tab): 
    if current_tab != 'tab-map':
        raise PreventUpdate
    
    map_df = get_global_data()
    map_fig = px.choropleth(map_df, 
                           locations="Country", 
                           locationmode='country names',
                           color="Cases", 
                           hover_name="Country",
                           hover_data=["Cases"],
                           color_continuous_scale="reds",
                           title="Global COVID-19 Cases Distribution")
    
    map_fig.update_layout(geo=dict(showframe=False, showcoastlines=False))
    return map_fig

# Callback for Forecast Table
@app.callback(
    Output('forecast-table', 'figure'),
    [Input('country_dd', 'value'),
    Input('tabs', 'value')]
)
def update_forecast_table(selected_country,current_tab):
	if current_tab != 'tab-table':
		raise PreventUpdate
	prophet_df, future_dates, forecast_values,prophet_metrics = prophet_forecast(selected_country)
	forecast_df = pd.DataFrame({
		'Date': future_dates,
		'Predicted Cases': forecast_values.round().astype(int)
	})

	table_fig = go.Figure(data=[go.Table(
		header=dict(values=['Date', 'Predicted Cases'], fill_color='paleturquoise', align='left'),
		cells=dict(values=[forecast_df['Date'].dt.strftime('%Y-%m-%d'), forecast_df['Predicted Cases']],
					fill_color='lavender', align='left'))
	])
	table_fig.update_layout(title=f'30-Day Forecast for {selected_country}')
	return table_fig

# Callback for Insights - RETURN A DATATABLE
@app.callback(
	Output('model-metrics', 'children'),
	[Input('country_dd', 'value'),
		Input('tabs', 'value')]
	)
def update_insights(selected_country, current_tab):
	if current_tab != 'tab-insights':
		raise PreventUpdate

	historical_df = get_country_data(selected_country)
	prophet_df, future_dates, forecast_values,prophet_metrics = prophet_forecast(selected_country)

	# Calculate metrics
	last_historical_value = historical_df['Cases'].iloc[-1]
	forecast_value_30d = forecast_values.iloc[-1] if len(forecast_values) > 0 else 0
	forecast_growth = ((forecast_value_30d - last_historical_value) / last_historical_value * 100) if last_historical_value > 0 else 0

	# Current trend analysis
	current_ma = historical_df['7Day_MA'].iloc[-1]
	previous_ma = historical_df['7Day_MA'].iloc[-8] if len(historical_df) > 7 else current_ma
	trend = "Increasing" if current_ma > previous_ma else "Decreasing"
	trend_percentage = abs((current_ma - previous_ma) / previous_ma * 100) if previous_ma > 0 else 0

	# Create data for table
	insights_data = [
		{"Metric": "Current Cases", "Value": f"{last_historical_value:,.0f}", "Description": "Latest available case count"},
		{"Metric": "30-Day Forecast", "Value": f"{forecast_value_30d:,.0f}", "Description": "Projected cases in 30 days"},
		{"Metric": "Projected Growth", "Value": f"{forecast_growth:+.1f}%", "Description": "Expected growth rate"},
		{"Metric": "7-Day Trend", "Value": trend, "Description": f"Current trend ({trend_percentage:.1f}% change)"},
		{"Metric": "Peak Cases", "Value": f"{historical_df['Cases'].max():,.0f}", "Description": "Highest recorded cases"},
		{"Metric": "Data Duration", "Value": f"{len(historical_df)} days", "Description": "Available historical data"}
	]

	insights_df = pd.DataFrame(insights_data)

	return html.Div([
		html.H4("📊 COVID-19 Insights", style={'textAlign': 'center', 'marginBottom': '20px'}),
		dash_table.DataTable(
			data=insights_df.to_dict('records'),
			columns=[{"name": i, "id": i} for i in insights_df.columns],
			style_cell={
				'textAlign': 'left',
				'padding': '10px',
				'fontFamily': 'Arial'
			},
			style_header={
				'backgroundColor': '#2c3e50',
				'color': 'white',
				'fontWeight': 'bold'
			},
			style_data={
				'backgroundColor': '#ecf0f1',
				'color': '#2c3e50'
			},
			style_data_conditional=[
				{
					'if': {'row_index': 'odd'},
					'backgroundColor': '#d6dbdf'
				}
			]
		)
	])
# callback for comparison model
@app.callback(
	[Output('model-comparison-plot','figure'),
	Output('model-metrics-comparison','children')],
	[Input('country_dd', 'value'),
	Input('tabs', 'value')]
)
def update_model_comparison(selected_country, current_tab):
    if current_tab != 'tab-comparison':
        raise PreventUpdate
    
    # Get both forecasts
    comparison = get_model_comparison(selected_country)
    
    # Create comparison plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison['historical']['Date'], 
        y=comparison['historical']['Cases'],
        mode='lines', 
        name='Historical Data', 
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=comparison['prophet']['dates'], 
        y=comparison['prophet']['values'],
        mode='lines+markers', 
        name='Prophet Forecast', 
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=comparison['random_forest']['dates'], 
        y=comparison['random_forest']['values'],
        mode='lines+markers', 
        name='Random Forest Forecast', 
        line=dict(color='green', dash='dot')
    ))
    
    fig.update_layout(
        title=f'Model Comparison for {selected_country}',
        xaxis_title='Date',
        yaxis_title='Cases',
        hovermode='x unified'
    )
    
    # Create metrics comparison
    metrics_html = html.Div([
        html.H4("📊 Model Performance Metrics:"),
        html.H5("🤖 Random Forest:"),
        html.P(f"• Mean Absolute Error (MAE): {comparison['random_forest']['metrics']['mae']:.2f}"),
        html.P(f"• Root Mean Squared Error (RMSE): {comparison['random_forest']['metrics']['rmse']:.2f}"),
        html.H5("🔮 Prophet:"),
		html.P(f"• Mean Absolute Error (MAE): {comparison['prophet']['metrics']['mae']:.2f}"),
        html.P(f"• Root Mean Squared Error (RMSE): {comparison['prophet']['metrics']['rmse']:.2f}"),
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    
    return fig, metrics_html
# Run the server
if __name__ == '__main__':
    app.run(debug=True)