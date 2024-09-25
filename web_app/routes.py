from flask import Blueprint, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
import json
import base64
from google.oauth2 import service_account
from google.cloud import storage

# Create the blueprint for the routes
main = Blueprint('main', __name__)

# Define bucket name and paths for GCS
bucket_name = 'weather-aqi-data-storage'
model_file_path = 'models/trained_model.pkl'
historical_data_file_path = 'merged_weather_aqi_2014_2024.csv'  # Updated path

# Function to download a file from GCS
def download_file_from_gcs(bucket_name, file_path):
    # Check if the credentials are in base64 and decode if necessary
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    try:
        # If the credentials are base64-encoded, decode them
        if credentials_json.startswith('{'):
            credentials_info = json.loads(credentials_json)
        else:
            credentials_info = json.loads(base64.b64decode(credentials_json).decode('utf-8'))
    except Exception as e:
        print(f"Error decoding credentials: {e}")
        raise

    # Load credentials and initialize the GCS client
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = storage.Client(credentials=credentials)

    # Fetch file from GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    return data

# Load the pre-trained model from GCS
def load_model_from_gcs(bucket_name, model_file_path):
    model_data = download_file_from_gcs(bucket_name, model_file_path)
    model = joblib.load(io.BytesIO(model_data))
    return model

# Load historical data from GCS
def load_historical_data_from_gcs(bucket_name, historical_data_file_path):
    data = download_file_from_gcs(bucket_name, historical_data_file_path)
    historical_data = pd.read_csv(io.StringIO(data.decode('utf-8')))
    historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
    historical_data['day_of_year'] = historical_data['datetime'].dt.dayofyear
    return historical_data

# Load the model and historical data
model = load_model_from_gcs(bucket_name, model_file_path)
historical_data = load_historical_data_from_gcs(bucket_name, historical_data_file_path)

# Function to prepare data for a given date
def prepare_input_data(selected_date):
    date_obj = datetime.strptime(selected_date, '%Y-%m-%d')

    # Get the day of year for the selected date
    day_of_year = date_obj.timetuple().tm_yday

    # Filter historical data to get the climate data for the same day of year in the past
    historical_day_data = historical_data[historical_data['day_of_year'] == day_of_year]

    # Ensure only numeric columns are used for mean calculation
    numeric_columns = historical_day_data.select_dtypes(include=[np.number])

    # If no valid data remains after cleaning, use global averages
    if numeric_columns.empty:
        print(f"No valid historical data found for day of year {day_of_year}. Using global averages.")
        numeric_columns = historical_data.select_dtypes(include=[np.number])

    # Calculate mean values for all relevant features
    mean_values = numeric_columns.mean()

    # Function to safely get the mean value for a column, with a default if it doesn't exist
    def get_mean_value(column_name, default_value=0):
        return mean_values.get(column_name, default_value)

    # Use historical AQI values for the lag features, or fall back to the mean AQI
    aqi_mean = historical_data['AQI Value'].apply(pd.to_numeric, errors='coerce').dropna().mean()

    # Prepare the input data based on the simplified model's expected features
    input_data = pd.DataFrame([[
        get_mean_value('temp_mean'),
        get_mean_value('temp_max_mean'),
        get_mean_value('temp_min_mean'),
        get_mean_value('wind_speed_mean'),
        get_mean_value('humidity_mean'),
        get_mean_value('pressure_mean'),
        get_mean_value('clouds_all_mean'),
        date_obj.month % 12 // 3 + 1,  # season
        day_of_year,
        date_obj.weekday(),
        1 if date_obj.weekday() >= 5 else 0,  # is_weekend
        get_mean_value('AQI_lag_1', aqi_mean),  # AQI lags
        get_mean_value('AQI_lag_3', aqi_mean),
        get_mean_value('temp_mean_7d_avg'),
        get_mean_value('humidity_mean_7d_avg'),
        get_mean_value('temp_mean') ** 2  # temp_mean_squared
    ]], columns=model.feature_names_in_)  # Ensure the same order and names as model expects

    return input_data

# Route for home page
@main.route('/')
def home():
    return render_template('index.html')

# Route to predict AQI for a selected date
@main.route('/predict', methods=['POST'])
def predict():
    selected_date = request.form['selected_date']
    input_data = prepare_input_data(selected_date)
    prediction = model.predict(input_data)[0]

    # Convert the prediction to a Python float before returning it as JSON
    prediction = float(prediction)

    return jsonify({'aqi_prediction': prediction})