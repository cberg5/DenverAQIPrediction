import pandas as pd
from google.cloud import storage
import io

# Function to load and process weather data from Google Cloud Storage
def load_and_process_weather_data(bucket_name, input_blob_name, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Download the weather data file from the bucket
    blob = bucket.blob(input_blob_name)
    data = blob.download_as_string()

    # Load the weather data into a pandas DataFrame
    weather_df = pd.read_csv(io.StringIO(data.decode('utf-8')))

    # Convert the 'dt_iso' column to a datetime object with the specified format
    weather_df['datetime'] = pd.to_datetime(weather_df['dt_iso'], format='%Y-%m-%d %H:%M:%S %z UTC')

    # Handle missing values (example: forward fill)
    weather_df.ffill(inplace=True)

    # Select only numeric columns for resampling
    numeric_cols = weather_df.select_dtypes(include=['number']).columns
    numeric_weather_df = weather_df[['datetime'] + list(numeric_cols)]

    # Resample to daily frequency and calculate mean, max, min, and standard deviation values for each day
    daily_mean = numeric_weather_df.resample('D', on='datetime').mean().add_suffix('_mean')
    daily_max = numeric_weather_df.resample('D', on='datetime').max().add_suffix('_max')
    daily_min = numeric_weather_df.resample('D', on='datetime').min().add_suffix('_min')
    daily_std = numeric_weather_df.resample('D', on='datetime').std().add_suffix('_std')

    # Combine the features into one DataFrame
    daily_weather_df = pd.concat([daily_mean, daily_max, daily_min, daily_std], axis=1).reset_index()

    # Convert the processed DataFrame back to CSV format
    processed_data = daily_weather_df.to_csv(index=False)

    # Save the processed daily weather data back to Google Cloud Storage
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(processed_data)

    print(f"Processed weather data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    # Define the Google Cloud Storage bucket and file paths
    bucket_name = 'weather-aqi-data-storage'  # Your GCS bucket name
    input_blob_name = 'weather/denver_weather_2014_2024.csv'  # Path to the input file in GCS
    output_blob_name = 'daily_denver_weather_2014_2024.csv'  # Path for the output file in GCS

    # Load and process the weather data, and save the result back to GCS
    load_and_process_weather_data(bucket_name, input_blob_name, output_blob_name)