import pandas as pd
from google.cloud import storage
import io

def load_and_process_weather_data(bucket_name, input_blob_name, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(input_blob_name)
    data = blob.download_as_string()

    weather_df = pd.read_csv(io.StringIO(data.decode('utf-8')))

    weather_df['datetime'] = pd.to_datetime(weather_df['dt_iso'], format='%Y-%m-%d %H:%M:%S %z UTC')

    weather_df.ffill(inplace=True)

    numeric_cols = weather_df.select_dtypes(include=['number']).columns
    numeric_weather_df = weather_df[['datetime'] + list(numeric_cols)]

    daily_mean = numeric_weather_df.resample('D', on='datetime').mean().add_suffix('_mean')
    daily_max = numeric_weather_df.resample('D', on='datetime').max().add_suffix('_max')
    daily_min = numeric_weather_df.resample('D', on='datetime').min().add_suffix('_min')
    daily_std = numeric_weather_df.resample('D', on='datetime').std().add_suffix('_std')

    daily_weather_df = pd.concat([daily_mean, daily_max, daily_min, daily_std], axis=1).reset_index()

    processed_data = daily_weather_df.to_csv(index=False)

    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(processed_data)

    print(f"Processed weather data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    input_blob_name = 'weather/denver_weather_2014_2024.csv'
    output_blob_name = 'daily_denver_weather_2014_2024.csv'

    load_and_process_weather_data(bucket_name, input_blob_name, output_blob_name)