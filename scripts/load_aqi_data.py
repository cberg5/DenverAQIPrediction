import glob
import pandas as pd

def load_and_combine_aqi_data(input_directory, output_path):
    # List all AQI files in the directory
    aqi_files = glob.glob(f'{input_directory}/denver_aqi_*.csv')

    # Load and concatenate AQI data
    aqi_dfs = []
    for file in aqi_files:
        df = pd.read_csv(file)

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

    # Save the combined and sorted AQI data
    aqi_df.to_csv(output_path, index=False)
    print(f"Combined and sorted AQI data saved to: {output_path}")

if __name__ == "__main__":
    # Define file paths
    input_directory = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/aqi'
    output_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/combined_aqi_2014_2024.csv'

    # Load, combine, and sort the AQI data
    load_and_combine_aqi_data(input_directory, output_path)