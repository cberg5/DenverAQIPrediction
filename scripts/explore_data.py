import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import storage
from io import StringIO


def load_data_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_string()
    df = pd.read_csv(StringIO(data.decode('utf-8')))
    return df


def clean_non_numeric(df):
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')

    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']
    for var in variables_to_plot:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    df.dropna(subset=['AQI Value'] + variables_to_plot, inplace=True)
    return df


def plot_scatter(df):
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']

    for var in variables_to_plot:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=df[var], y=df['AQI Value'], scatter_kws={'alpha': 0.3}, line_kws={"color": "red"})
        plt.title(f'AQI vs {var}')
        plt.xlabel(var)
        plt.ylabel('AQI Value')
        plt.show()

def plot_aqi_over_time(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='datetime', y='AQI Value', data=df, color='blue')
    plt.title('AQI Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.show()

def explore_data(bucket_name, file_name):
    df = load_data_from_gcs(bucket_name, file_name)

    df['datetime'] = pd.to_datetime(df['datetime'])

    df = clean_non_numeric(df)

    plot_scatter(df)
    plot_aqi_over_time(df)


if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    file_name = 'merged_weather_aqi_2014_2024.csv'

    explore_data(bucket_name, file_name)
