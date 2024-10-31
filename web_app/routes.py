from flask import Blueprint, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
aqi_data_file_path = 'combined_aqi_2014_2024.csv'
weather_data_file_path = 'denver_weather_2014_2024.csv'
historical_data_file_path = 'merged_weather_aqi_2014_2024.csv'


def load_csv_from_gcs(bucket_name, file_path):
    data = download_file_from_gcs(bucket_name, file_path)
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df


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


def load_historical_data_from_gcs(bucket_name, historical_data_file_path):
    data = download_file_from_gcs(bucket_name, historical_data_file_path)
    historical_data = pd.read_csv(io.StringIO(data.decode('utf-8')))
    historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
    historical_data['day_of_year'] = historical_data['datetime'].dt.dayofyear
    return historical_data


model = load_model_from_gcs(bucket_name, model_file_path)
historical_data = load_historical_data_from_gcs(bucket_name, historical_data_file_path)


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


@main.route('/download_aqi_data', methods=['GET'])
def download_aqi_data():
    try:
        aqi_data = download_file_from_gcs(bucket_name, aqi_data_file_path)

        csv_stream = io.BytesIO(aqi_data)

        return send_file(csv_stream, mimetype='text/csv', as_attachment=True, download_name='aqi_data.csv')
    except Exception as e:
        print(f"Error downloading AQI data: {e}")
        return jsonify({"error": "Failed to download AQI data"}), 500


@main.route('/download_weather_data', methods=['GET'])
def download_weather_data():
    try:
        weather_data = download_file_from_gcs(bucket_name, weather_data_file_path)

        csv_stream = io.BytesIO(weather_data)

        return send_file(csv_stream, mimetype='text/csv', as_attachment=True, download_name='weather_data.csv')
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return jsonify({"error": "Failed to download weather data"}), 500

@main.route('/download_cleaned_data', methods=['GET'])
def download_cleaned_data():
    try:
        cleaned_data = download_file_from_gcs(bucket_name, historical_data_file_path)

        csv_stream = io.BytesIO(cleaned_data)

        return send_file(csv_stream, mimetype='text/csv', as_attachment=True, download_name='combined_cleaned_data.csv')
    except Exception as e:
        print(f"Error downloading combined cleaned data: {e}")
        return jsonify({"error": "Failed to download combined cleaned data"}), 500

@main.route('/plot_scatter', methods=['GET'])
def plot_scatter_route():
    try:
        df = load_csv_from_gcs(bucket_name, historical_data_file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])

        df = clean_non_numeric(df)

        img = plot_scatter(df)

        return send_file(img, mimetype='image/png')
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
        return jsonify({"error": "Failed to generate scatter plot"}), 500

@main.route('/plot_aqi_over_time', methods=['GET'])
def plot_aqi_over_time_route():
    try:
        df = load_csv_from_gcs(bucket_name, historical_data_file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])

        df = clean_non_numeric(df)

        img = plot_aqi_over_time(df)

        return send_file(img, mimetype='image/png')
    except Exception as e:
        print(f"Error generating AQI over time plot: {e}")
        return jsonify({"error": "Failed to generate AQI over time plot"}), 500


def clean_non_numeric(df):
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')

    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']
    for var in variables_to_plot:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    df.dropna(subset=['AQI Value'] + variables_to_plot, inplace=True)
    return df


def plot_scatter(df):
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes = axes.flatten()

    for idx, var in enumerate(variables_to_plot):
        sns.regplot(x=df[var], y=df['AQI Value'], scatter_kws={'alpha': 0.3}, line_kws={"color": "red"}, ax=axes[idx])
        axes[idx].set_title(f'AQI vs {var}')
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel('AQI Value')

    fig.delaxes(axes[-1])

    plt.tight_layout()

    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)

    return img_stream


def plot_aqi_over_time(df):
    img_stream = io.BytesIO()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='datetime', y='AQI Value', data=df, color='blue')
    plt.title('AQI Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')

    plt.savefig(img_stream, format='png')
    img_stream.seek(0)

    return img_stream
