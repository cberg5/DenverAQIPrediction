import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Function to clean non-numeric values
def clean_non_numeric(df):
    # Convert 'AQI Value' and relevant variables to numeric, coerce errors to NaN
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')

    # Ensure key weather variables are numeric
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']
    for var in variables_to_plot:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    # Drop rows with NaN values in these columns for valid plotting
    df.dropna(subset=['AQI Value'] + variables_to_plot, inplace=True)

    return df


# 1. Scatter Plot Function with Adjustments
def plot_scatter(df):
    variables_to_plot = ['temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean', 'clouds_all_mean']

    for var in variables_to_plot:
        plt.figure(figsize=(10, 6))

        # Use seaborn's regplot for scatter plot with a trend line
        sns.regplot(x=df[var], y=df['AQI Value'], scatter_kws={'alpha': 0.3}, line_kws={"color": "red"})

        # Add jitter to separate overlapping points
        jittered_values = df[var] + 0.2 * (np.random.rand(len(df)) - 0.5)
        plt.scatter(jittered_values, df['AQI Value'], alpha=0.4)

        plt.title(f'AQI vs {var}')
        plt.xlabel(var)
        plt.ylabel('AQI Value')

        # Set AQI limits for better readability (optional)
        plt.ylim(df['AQI Value'].min() - 10, df['AQI Value'].max() + 10)

        plt.show()


# 2. Time Series Plot Function
def plot_aqi_over_time(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='datetime', y='AQI Value', data=df, color='blue')
    plt.title('AQI Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.show()


# 3. Overlay AQI and Temperature Over Time
def plot_aqi_with_temperature(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('AQI', color='blue')
    ax1.plot(df['datetime'], df['AQI Value'], color='blue', label='AQI')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (Mean)', color='red')
    ax2.plot(df['datetime'], df['temp_mean'], color='red', label='Temperature')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('AQI and Temperature Over Time')
    fig.tight_layout()
    plt.show()


# 4. Seasonal Box Plot
def plot_aqi_by_season(df):
    df['season'] = df['datetime'].dt.month % 12 // 3 + 1  # Get seasons
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='season', y='AQI Value', data=df)
    plt.title('AQI by Season')
    plt.show()


# 5. Outliers via Box Plot
def plot_aqi_boxplot(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='AQI Value', data=df)
    plt.title('Boxplot of AQI Values')
    plt.show()


# 6. Correlation of Enhanced Features
def plot_enhanced_corr(df):
    # Clean non-numeric values in the dataset
    df = clean_non_numeric(df)

    # Fill NaN values to avoid errors when calculating correlations
    df.fillna(method='ffill', inplace=True)

    # Select only numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Generate the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix with Enhanced Features')
    plt.show()


# Main EDA Function
def explore_data(data_path):
    # Load data
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Clean data for numeric values
    df = clean_non_numeric(df)

    # Call individual functions for different analyses
    plot_scatter(df)  # Scatter plots of AQI vs key variables
    plot_aqi_over_time(df)  # Time series plot of AQI
    plot_aqi_with_temperature(df)  # Overlay AQI and temperature
    plot_aqi_by_season(df)  # Seasonal variation of AQI
    plot_aqi_boxplot(df)  # Boxplot of AQI to identify outliers
    plot_enhanced_corr(df)  # Correlation matrix with enhanced features


if __name__ == "__main__":
    # Define the file path to your dataset
    data_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/merged_weather_aqi_2014_2024.csv'
    explore_data(data_path)
