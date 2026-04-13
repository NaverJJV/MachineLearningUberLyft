import argparse
import pandas as pd


def load_and_preprocess(rides_path, weather_path):
    print(f"Loading rides data from {rides_path}...")
    rides_df = pd.read_csv(rides_path)

    print(f"Loading weather data from {weather_path}...")
    weather_df = pd.read_csv(weather_path)

    # --- Phase 1: Basic Cleaning ---
    print("Dropping rows with missing prices...")
    rides_df = rides_df.dropna(subset=['price'])

    print("Bucketing prices into 3 categories...")
    rides_df['price_category'] = pd.qcut(rides_df['price'], q=3, labels=['Low', 'Medium', 'High'])
    rides_df = rides_df.drop(columns=['price'])

    # --- Phase 2: Feature Engineering (Time & Weather Alignment) ---
    print("Aligning timestamps and merging weather data...")
    rides_df['datetime'] = pd.to_datetime(rides_df['time_stamp'], unit='ms')
    weather_df['datetime'] = pd.to_datetime(weather_df['time_stamp'], unit='s')

    rides_df['merge_hour'] = rides_df['datetime'].dt.floor('h')
    weather_df['merge_hour'] = weather_df['datetime'].dt.floor('h')

    weather_numeric = weather_df.drop(columns=['time_stamp', 'datetime'])
    weather_grouped = weather_numeric.groupby(['location', 'merge_hour']).mean().reset_index()

    merged_df = rides_df.merge(weather_grouped,
                               left_on=['source', 'merge_hour'],
                               right_on=['location', 'merge_hour'],
                               how='left')

    merged_df = merged_df.drop(columns=['time_stamp', 'location', 'merge_hour', 'datetime'])

    weather_cols = ['temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind']
    for col in weather_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

    # --- Phase 3: Encoding Categorical Variables ---
    print("Dropping useless columns...")
    merged_df = merged_df.drop(columns=['id', 'product_id'])

    print("Encoding the target variable...")
    category_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    merged_df['price_category'] = merged_df['price_category'].map(category_mapping)

    print("One-Hot Encoding categorical features...")
    categorical_cols = ['cab_type', 'destination', 'source', 'name']
    # drop_first=True prevents the "dummy variable trap" (multicollinearity)
    merged_df = pd.get_dummies(merged_df, columns=categorical_cols, drop_first=True)

    print("\nFinal Preprocessing complete. Final dataset shape:", merged_df.shape)

    # Save the ready-to-train data
    output_filename = "processed_data.csv"
    print(f"Saving processed data to '{output_filename}'...")
    merged_df.to_csv(output_filename, index=False)
    print("Done!")

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Uber and Lyft Cab Prices dataset.")
    parser.add_argument("--rides", type=str, required=True, help="Path to the cab_rides.csv file")
    parser.add_argument("--weather", type=str, required=True, help="Path to the weather.csv file")

    args = parser.parse_args()

    final_df = load_and_preprocess(args.rides, args.weather)