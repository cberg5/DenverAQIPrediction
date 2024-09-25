import pandas as pd
from google.cloud import storage
import io

# Function to merge weather and AQI data from Google Cloud Storage
def merge_weather_and_aqi(bucket_name, weather_blob_name, aqi_blob_name, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Download the weather data file from the bucket
    weather_blob = bucket.blob(weather_blob_name)
    weather_data = weather_blob.download_as_string()
    weather_df = pd.read_csv(io.StringIO(weather_data.decode('utf-8')))

    # Download the AQI data file from the bucket
    aqi_blob = bucket.blob(aqi_blob_name)
    aqi_data = aqi_blob.download_as_string()
    aqi_df = pd.read_csv(io.StringIO(aqi_data.decode('utf-8')))

    # Ensure both date columns are datetime objects
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)  # Remove timezone
    aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])  # Ensure it's datetime without timezone

    # Merge the datasets on the date columns
    merged_df = pd.merge(weather_df, aqi_df, left_on='datetime', right_on='Date', how='outer')

    # Handle missing values (e.g., forward fill)
    merged_df.fillna(method='ffill', inplace=True)

    # Drop unnecessary columns if needed
    merged_df.drop(columns=['Date'], inplace=True)

    # Convert the merged DataFrame to CSV format
    merged_data = merged_df.to_csv(index=False)

    # Save the merged dataset back to Google Cloud Storage
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(merged_data)

    print(f"Merged data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    # Define the Google Cloud Storage bucket and file paths
    bucket_name = 'weather-aqi-data-storage'  # Your GCS bucket name
    weather_blob_name = 'daily_denver_weather_2014_2024.csv'  # Weather data path in GCS
    aqi_blob_name = 'combined_aqi_2014_2024.csv'  # AQI data path in GCS
    output_blob_name = 'merged_weather_aqi_2014_2024.csv'  # Output path in GCS

    # Merge the weather and AQI datasets and save the result back to GCS
    merge_weather_and_aqi(bucket_name, weather_blob_name, aqi_blob_name, output_blob_name)