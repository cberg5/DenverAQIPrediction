import pandas as pd
from google.cloud import storage
import io

# Function to load and combine AQI data from Google Cloud Storage
def load_and_combine_aqi_data(bucket_name, input_prefix, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # List all AQI files in the bucket that start with the input_prefix (like 'denver_aqi_')
    blobs = bucket.list_blobs(prefix=input_prefix)

    aqi_dfs = []
    for blob in blobs:
        # Download the blob as a string and load it into a pandas DataFrame
        data = blob.download_as_string()
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))

        # Strip any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert the 'Date' column to datetime, coercing errors to NaT
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)

        aqi_dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    aqi_df = pd.concat(aqi_dfs, ignore_index=True)

    # Drop any rows where the Date couldn't be parsed
    aqi_df = aqi_df.dropna(subset=['Date'])

    # Sort the DataFrame by the 'Date' column
    aqi_df = aqi_df.sort_values(by='Date').reset_index(drop=True)

    # Save the combined and sorted AQI data to a new CSV in Google Cloud Storage
    combined_data = aqi_df.to_csv(index=False)

    # Upload the combined file back to Google Cloud Storage
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(combined_data)
    print(f"Combined and sorted AQI data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    # Define the bucket name and input/output paths in Google Cloud Storage
    bucket_name = 'weather-aqi-data-storage'  # Replace with your GCS bucket name
    input_prefix = 'aqi/denver_aqi_'  # The folder and file prefix for AQI files in the bucket
    output_blob_name = 'combined_aqi_2014_2024.csv'  # The output file name in the bucket

    # Load, combine, and sort the AQI data, and save the combined file to GCS
    load_and_combine_aqi_data(bucket_name, input_prefix, output_blob_name)