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

main = Blueprint('main', __name__)

bucket_name = 'weather-aqi-data-storage'
model_file_path = 'models/trained_model.pkl'
aqi_data_file_path = 'combined_aqi_2014_2024.csv'  # AQI data file in GCS
weather_data_file_path = 'weather/denver_weather_2014_2024.csv'  # Denver weather data file in GCS
historical_data_file_path = 'merged_weather_aqi_2014_2024.csv'  # Historical data for prediction

def download_file_from_gcs(bucket_name, file_path):
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    try:
        if credentials_json.startswith('{'):
            credentials_info = json.loads(credentials_json)
        else:
            credentials_info = json.loads(base64.b64decode(credentials_json).decode('utf-8'))
    except Exception as e:
        print(f"Error decoding credentials: {e}")
        raise

    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = storage.Client(credentials=credentials)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    return data

def load_model_from_gcs(bucket_name, model_file_path):
    model_data = download_file_from_gcs(bucket_name, model_file_path)
    model = joblib.load(io.BytesIO(model_data))
    return model

def load_csv_from_gcs(bucket_name, file_path):
    data = download_file_from_gcs(bucket_name, file_path)
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

model = load_model_from_gcs(bucket_name, model_file_path)
historical_data = load_csv_from_gcs(bucket_name, historical_data_file_path)

def prepare_input_data(selected_date):
    date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday
    historical_day_data = historical_data[historical_data['day_of_year'] == day_of_year]

    numeric_columns = historical_day_data.select_dtypes(include=[np.number])

    if numeric_columns.empty:
        print(f"No valid historical data found for day of year {day_of_year}. Using global averages.")
        numeric_columns = historical_data.select_dtypes(include=[np.number])

    mean_values = numeric_columns.mean()

    def get_mean_value(column_name, default_value=0):
        return mean_values.get(column_name, default_value)

    aqi_mean = historical_data['AQI Value'].apply(pd.to_numeric, errors='coerce').dropna().mean()

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

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    selected_date = request.form['selected_date']
    input_data = prepare_input_data(selected_date)
    prediction = model.predict(input_data)[0]

    prediction = float(prediction)

    return jsonify({'aqi_prediction': prediction})

# Route to load AQI data from GCS
@main.route('/load_aqi_data', methods=['GET'])
def load_aqi_data():
    aqi_data = load_csv_from_gcs(bucket_name, aqi_data_file_path)
    aqi_data_json = aqi_data.head(100).to_dict(orient='records')  # Limit to 100 rows for simplicity
    return jsonify(aqi_data_json)

# Route to load Denver weather data from GCS
@main.route('/load_weather_data', methods=['GET'])
def load_weather_data():
    weather_data = load_csv_from_gcs(bucket_name, weather_data_file_path)
    weather_data_json = weather_data.head(100).to_dict(orient='records')  # Limit to 100 rows for simplicity
    return jsonify(weather_data_json)