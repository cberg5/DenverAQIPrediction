import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from google.cloud import storage
from io import StringIO

def load_data_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_string()
    df = pd.read_csv(StringIO(data.decode('utf-8')))
    return df

def explore_data(bucket_name, file_name):

    df = load_data_from_gcs(bucket_name, file_name)

    if 'datetime' not in df.columns:
        raise KeyError("The 'datetime' column is missing from the DataFrame.")

    df['datetime'] = pd.to_datetime(df['datetime'])

    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')

    df['AQI Value'] = df['AQI Value'].ffill()

    df['AQI_7d_avg'] = df['AQI Value'].rolling(window=7).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['AQI_7d_avg'], label='7-Day Rolling Average AQI', color='blue')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('7-Day Rolling Average AQI Over Time')
    plt.legend()
    plt.show()

    numeric_columns = df.select_dtypes(include=['number']).columns
    df_weekly = df[['datetime'] + list(numeric_columns)].resample('W', on='datetime').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_weekly.index, df_weekly['AQI Value'], label='Weekly Average AQI', color='green')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Weekly Average AQI Over Time')
    plt.legend()
    plt.show()

    fig = px.line(df, x='datetime', y='AQI Value', title='AQI Over Time (Interactive)')
    fig.update_traces(mode='lines+markers')
    fig.show()

    date_start = '2023-01-01'
    date_end = '2023-06-30'
    df_subset = df[(df['datetime'] >= date_start) & (df['datetime'] <= date_end)]
    plt.figure(figsize=(12, 6))
    plt.plot(df_subset['datetime'], df_subset['AQI Value'], label='AQI Value', color='purple')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title(f'AQI from {date_start} to {date_end}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['AQI Value'], label='AQI Value', color='red')

    important_dates = {
        '2022-01-01': "New Year's Day",
        '2023-07-04': 'Independence Day'
    }
    for date, event in important_dates.items():
        plt.annotate(event, xy=(pd.to_datetime(date), df.loc[df['datetime'] == date, 'AQI Value'].values[0]),
                     xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))

    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('AQI Over Time with Annotations')
    plt.legend()
    plt.show()

    specific_date = '2023-01-01'
    specific_day_data = df[df['datetime'] == specific_date]
    print(f"Summary for {specific_date}:")
    print(specific_day_data.describe())

    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)

    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(15, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    key_columns = [
        'AQI Value', 'temp_mean', 'temp_max', 'temp_min',
        'humidity_mean', 'humidity_max', 'humidity_min',
        'wind_speed_mean', 'wind_gust_mean',
        'pressure_mean', 'pressure_std',
        'rain_1h_mean', 'rain_3h_mean',
        'snow_1h_mean', 'snow_3h_mean',
        'clouds_all_mean'
    ]
    df_subset = df[key_columns]

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_subset.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix (Key Variables)')
    plt.show()

if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    file_name = 'merged_weather_aqi_2014_2024.csv'

    explore_data(bucket_name, file_name)
