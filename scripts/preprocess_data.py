import pandas as pd
from google.cloud import storage
import io

def merge_weather_and_aqi(bucket_name, weather_blob_name, aqi_blob_name, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    weather_blob = bucket.blob(weather_blob_name)
    weather_data = weather_blob.download_as_string()
    weather_df = pd.read_csv(io.StringIO(weather_data.decode('utf-8')))

    aqi_blob = bucket.blob(aqi_blob_name)
    aqi_data = aqi_blob.download_as_string()
    aqi_df = pd.read_csv(io.StringIO(aqi_data.decode('utf-8')))

    weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)  # Remove timezone
    aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])  # Ensure it's datetime without timezone

    merged_df = pd.merge(weather_df, aqi_df, left_on='datetime', right_on='Date', how='outer')

    merged_df.fillna(method='ffill', inplace=True)

    merged_df.drop(columns=['Date'], inplace=True)

    merged_data = merged_df.to_csv(index=False)

    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(merged_data)

    print(f"Merged data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    weather_blob_name = 'daily_denver_weather_2014_2024.csv'
    aqi_blob_name = 'combined_aqi_2014_2024.csv'
    output_blob_name = 'merged_weather_aqi_2014_2024.csv'

    merge_weather_and_aqi(bucket_name, weather_blob_name, aqi_blob_name, output_blob_name)