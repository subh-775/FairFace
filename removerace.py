#  The output csv will have 5 more rows due to original data you can simply remove them using this code 
import argparse
import pandas as pd
import os

def filter_and_clean_csv(csv_path):
    # Read the CSV
    df = pd.read_csv(csv_path, header=None)
    
    # Filter rows starting with "detected_faces/race"
    mask = df[0].str.startswith("detected_faces/race")
    race_rows = df[mask]

    if race_rows.empty:
        print("No rows with pattern 'detected_faces/race' found.")
        return

    print("Entries starting with 'detected_faces/race':\n")
    print(race_rows.to_string(index=False))

    user_input = input("\nDo you want to remove these entries? (y/n): ").strip().lower()
    if user_input == 'y':
        cleaned_df = df[~mask]
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_clean{ext}"
        cleaned_df.to_csv(output_path, index=False, header=False)
        print(f"\nCleaned CSV saved as: {output_path}")
    else:
        print("\nNo changes made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'detected_faces/race...' entries from CSV.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    filter_and_clean_csv(args.csv)
