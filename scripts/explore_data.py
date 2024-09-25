import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage
from io import StringIO


# Function to load data from Google Cloud Storage
def load_data_from_gcs(bucket_name, file_name):
    client = storage.Client()  # This assumes your GOOGLE_APPLICATION_CREDENTIALS are set
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_string()  # Download the file as a string
    df = pd.read_csv(StringIO(data.decode('utf-8')))  # Read the CSV data into a pandas DataFrame
    return df


# Function to clean non-numeric values
def clean_non_numeric(df):
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')

    # Ensure key weather variables are numeric
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']
    for var in variables_to_plot:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    # Drop rows with NaN values in these columns for valid plotting
    df.dropna(subset=['AQI Value'] + variables_to_plot, inplace=True)
    return df


# 1. Scatter Plot Function
def plot_scatter(df):
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']

    for var in variables_to_plot:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=df[var], y=df['AQI Value'], scatter_kws={'alpha': 0.3}, line_kws={"color": "red"})
        plt.title(f'AQI vs {var}')
        plt.xlabel(var)
        plt.ylabel('AQI Value')
        plt.show()


# Other plot functions remain unchanged...
# 2. Time Series Plot Function
def plot_aqi_over_time(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='datetime', y='AQI Value', data=df, color='blue')
    plt.title('AQI Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.show()


# Main EDA function
def explore_data(bucket_name, file_name):
    # Load data from Google Cloud Storage
    df = load_data_from_gcs(bucket_name, file_name)

    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Clean non-numeric data
    df = clean_non_numeric(df)

    # Call individual analysis functions
    plot_scatter(df)
    plot_aqi_over_time(df)


if __name__ == "__main__":
    # Specify your Google Cloud Storage bucket and file
    bucket_name = 'weather-aqi-data-storage'  # Your GCS bucket name
    file_name = 'merged_weather_aqi_2014_2024.csv'  # The dataset file name in GCS

    # Explore the data
    explore_data(bucket_name, file_name)
