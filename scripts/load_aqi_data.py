import pandas as pd
from google.cloud import storage
import io


def load_and_combine_aqi_data(bucket_name, input_prefix, output_blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_prefix)

    aqi_dfs = []
    for blob in blobs:
        data = blob.download_as_string()
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))

        df.columns = df.columns.str.strip()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)

        aqi_dfs.append(df)

    aqi_df = pd.concat(aqi_dfs, ignore_index=True)

    aqi_df = aqi_df.dropna(subset=['Date'])

    aqi_df = aqi_df.sort_values(by='Date').reset_index(drop=True)

    combined_data = aqi_df.to_csv(index=False)

    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(combined_data)
    print(f"Combined and sorted AQI data saved to: gs://{bucket_name}/{output_blob_name}")

if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    input_prefix = 'aqi/denver_aqi_'
    output_blob_name = 'combined_aqi_2014_2024.csv'

    load_and_combine_aqi_data(bucket_name, input_prefix, output_blob_name)