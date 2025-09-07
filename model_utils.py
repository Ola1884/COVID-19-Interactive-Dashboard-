import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
# 1. LOAD AND CLEAN DATA ONCE, WHEN THE MODULE IS IMPORTED
data_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

def load_and_clean_data():
    """
    Loads and cleans the COVID-19 data 
    """
    print("Loading and cleaning global data...")

    # Load the data - NO NEED for error handling since she cleans it properly
    df = pd.read_csv(data_link)
    print(f"Raw data loaded. Shape: {df.shape}")  # DEBUG
    # YOUR TEAMMATE'S SUPERIOR CLEANING CODE:
    # Drop unnecessary columns
    df_clean = df.drop(columns=['Lat', 'Long'])

    # Melt the dataframe to convert from wide to long format
    df_clean = df_clean.melt(id_vars=['Province/State', 'Country/Region'], 
                                var_name='Date', value_name='Cases')

    # Convert to datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])

    # --- CRITICAL: ADD YOUR TEAMMATE'S DATA CLEANING STEPS ---
    # Clean and convert Cases column (handles spaces, nulls, etc.)
    df_clean['Cases'] = pd.to_numeric(df_clean['Cases'].astype(str).str.replace(' ',''), errors='coerce')
    df_clean['Cases'] = df_clean.groupby('Country/Region')['Cases'].ffill().fillna(0)
    # --- END OF CRITICAL CLEANING STEPS ---

    # Aggregate by country (summing all provinces)
    df_clean = df_clean.groupby(['Country/Region', 'Date'], as_index=False)['Cases'].sum()

    # Rename for consistency
    df_clean.rename(columns={'Country/Region': 'Country'}, inplace=True)

    # Calculate 7-day moving average
    df_clean['7Day_MA'] = df_clean.groupby('Country')['Cases'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Calculate daily new cases (from your teammate's notebook)
    df_clean['NewCases'] = df_clean.groupby('Country')['Cases'].diff().fillna(0)
    print(f"Cleaned data shape: {df_clean.shape}")  # DEBUG
    print(f"Available countries: {df_clean['Country'].nunique()}")  # DEBUG
    return df_clean

# Load data once when module is imported
GLOBAL_DF = load_and_clean_data()

def get_available_countries():
    """Returns list of available countries for dropdown."""
    return sorted(GLOBAL_DF['Country'].unique())

def get_country_data(country_name):
    country_data = GLOBAL_DF[GLOBAL_DF['Country'] == country_name].copy()
    return country_data[['Date', 'Cases', '7Day_MA','NewCases']]
def get_global_data():
    """Gets the latest case count for every country for the map."""
    latest_date = GLOBAL_DF['Date'].max()
    latest_data = GLOBAL_DF[GLOBAL_DF['Date'] == latest_date]
    return latest_data

def prophet_forecast(country_name, periods=30):
    """
    Generates forecast using Prophet model for specified country.
    Returns: historical data, forecast dates, forecast values
    """
    # 1. Get the data for the specific country (using the cleaned data)
    df_country = get_country_data(country_name)

    # 2. PREPARE DATA FOR PROPHET: Rename columns to 'ds' and 'y'
    df_prophet = df_country[['Date', 'Cases']].rename(columns={'Date': 'ds', 'Cases': 'y'})

    # 3. Create and fit model (using parameters from notebook)
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)
    # 4. Make a future dataframe & predict
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    # --- CALCULATE METRICS ---
    # Get the historical period from forecast
    historical_forecast = forecast[forecast['ds'].isin(df_prophet['ds'])]
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(df_prophet['y'], historical_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(df_prophet['y'], historical_forecast['yhat']))
    
    print(f"   Prophet Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return df_prophet, forecast['ds'].tail(periods), forecast['yhat'].tail(periods), {'mae': mae, 'rmse': rmse}

def random_forest_forecast(country_name, forecast_days=14, lookback_days=30):
    """
    Random Forest forecast adapted to available data
    """
    print(f"🌳 RANDOM FOREST for: {country_name}")
    
    # Get country data
    country_data = get_country_data(country_name)
    total_days = len(country_data)
    print(f"   Available data: {total_days} days")
    
    # Dynamically adjust based on available data
    if total_days < 44:  # 30 + 14
        # Reduce requirements further for countries with very little data
        lookback_days = min(20, total_days - forecast_days)
        forecast_days = min(7, total_days - lookback_days)
        print(f"   Adjusted: lookback={lookback_days}, forecast={forecast_days}")
    
    if total_days < lookback_days + forecast_days:
        print(f"   ⚠️  Still not enough data. Need {lookback_days + forecast_days}, have {total_days}")
        # Return empty but properly structured data
        last_date = country_data['Date'].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        return future_dates, [country_data['Cases'].iloc[-1]] * forecast_days, {'mae': 0, 'rmse': 0}
    
    # --- DATA PREPARATION ---
    cases = country_data['Cases'].values
    
    # INITIALIZE X AND Y ARRAYS
    X, y = [], []
    
    for i in range(len(cases) - lookback_days - forecast_days + 1):
        X.append(cases[i:i + lookback_days])
        y.append(cases[i + lookback_days:i + lookback_days + forecast_days])

    X = np.array(X)
    y = np.array(y)
    
    print(f"   Created {len(X)} training sequences")
    
    # ... rest of your function remains the same ...
    # --- TRAIN/TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # --- MULTI-OUTPUT RANDOM FOREST ---
    print("   Training Multi-Output Random Forest...")
    
    # Create base model
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Wrap with MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # --- MODEL EVALUATION ---
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each forecast day
    mae_scores = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(forecast_days)]
    rmse_scores = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(forecast_days)]

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    
    # Use the most recent data for forecasting
    last_sequence = cases[-lookback_days:]
    future_predictions = model.predict(last_sequence.reshape(1, -1))[0]
    
    # Generate future dates
    last_date = country_data['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
    
    return future_dates, future_predictions, {'mae': avg_mae, 'rmse': avg_rmse} 
def get_model_comparison(country_name):
    """
    Compare Prophet vs Random Forest performance
    """
    # Get historical data
    historical_df = get_country_data(country_name)

    # Prophet forecast
    prophet_df, prophet_dates, prophet_values, prophet_metrics = prophet_forecast(country_name)
    
    # Convert Prophet outputs to lists for consistency
    prophet_dates_list = prophet_dates.tolist() if hasattr(prophet_dates, 'tolist') else list(prophet_dates)
    prophet_values_list = prophet_values.tolist() if hasattr(prophet_values, 'tolist') else list(prophet_values)

    # Random Forest forecast
    rf_dates, rf_values, rf_metrics = random_forest_forecast(country_name)
    
    # Ensure RF outputs are lists
    rf_dates_list = list(rf_dates) if not isinstance(rf_dates, list) else rf_dates
    rf_values_list = list(rf_values) if not isinstance(rf_values, list) else rf_values

    return {
        'historical': historical_df,
        'prophet': {'dates': prophet_dates_list, 'values': prophet_values_list, 'metrics': prophet_metrics},
        'random_forest': {'dates': rf_dates_list, 'values': rf_values_list, 'metrics': rf_metrics}
    }
# Load data once when module is imported
cleaned_data = load_and_clean_data()
available_countries = get_available_countries()