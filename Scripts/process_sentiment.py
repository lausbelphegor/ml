"""
Sentiment Data Processing Script

This script loads raw sentiment data from a JSON file, applies multipliers to sentiment scores,
calculates various means for each day's data, and saves the processed data back to a JSON file.

Author: og
Date: 2023-08-11
"""
import csv
import configparser
import json
from datetime import timedelta, datetime
import numpy as np
import helper_functions as hf

hf.change_working_directory()

config = configparser.ConfigParser()
config.read('config.ini')

TEST_MODE = config['Settings']['testmode'] == 'True'

INPUT_FILE = config['Paths']['inputprocessing'] if not TEST_MODE else config['Paths']['TEST_inputprocessing']
OUTPUT_FILE = config['Paths']['outputprocessing'] if not TEST_MODE else config['Paths']['TEST_outputprocessing']
OUTPUT_CSV_FILE = config['Paths']['outputCSVprocessing'] if not TEST_MODE else config['Paths']['TEST_outputCSVprocessing']

NEG_MULTIPLIER = 1
NEUT_MULTIPLIER = 1
POS_MULTIPLIER = 1

MULTIPLIERS = {
    -1: NEG_MULTIPLIER,
    0: NEUT_MULTIPLIER,
    1: POS_MULTIPLIER
}


def load_data(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(file_path, 'r', encoding="utf-8") as json_file:
        return json.load(json_file)


def save_data(data, file_path):
    """
    Save data to a JSON file.

    Args:
        data (dict): Data to be saved.
        file_path (str): Path to the target JSON file.
    """
    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def apply_multiplier(value, category, multipliers):
    """
    Apply a multiplier to a value based on its category.

    Args:
        value (float): Value to be multiplied.
        category (int): Category (-1, 0, 1) to determine the multiplier.
        multipliers (dict): Multipliers for each category.

    Returns:
        float: Multiplied value.
    """
    # Default to 1.0 if category is not found
    multiplier = multipliers.get(category, 1.0)
    return value * multiplier


def arithmetic_mean(data):
    """
    Calculate the arithmetic mean of a list of values.

    Args:
        data (list): List of numerical values.

    Returns:
        float: Arithmetic mean.
    """
    return np.mean(data)


def process_day(day_data, multipliers):
    """
    Process sentiment data for a day, applying multipliers and calculating mean.

    Args:
        day_data (list): List of [label, score] pairs for a day.
        multipliers (dict): Multipliers for each sentiment category.

    Returns:
        tuple: Tuple containing processed day data, arithmetic mean
    """
    processed_day = []
    for item in day_data:
        category, value = item
        # Before -> apply_multiplier(value, category, multipliers) which was mlutiplying confidence values
        processed_value = apply_multiplier(category, category, multipliers)
        processed_day.append(processed_value)

    arith_mean = arithmetic_mean(processed_day)

    return processed_day, arith_mean


def process_data(data, multipliers):
    """
    Process sentiment data for all days, including means.

    Args:
        data (dict): Raw sentiment data.
        multipliers (dict): Multipliers for each sentiment category.

    Returns:
        dict: Processed sentiment data with means.
    """
    processed_data = {}
    for date, day_data in data.items():
        processed_day, arith_mean = process_day(
            day_data, multipliers)
        processed_data[date] = {
            'processed_values': processed_day,
            'arithmetic_mean': arith_mean
        }
    return processed_data


def write_csv(data):
    """
    Write sentiment means to a CSV file, filling in missing dates.

    Args:
        data (dict): Processed sentiment data containing date-wise means.

    Writes a CSV file named 'sentiment_means.csv' containing a chronological list
    of dates along with their corresponding sentiment means. If sentiment data is
    not available for a specific date, a default value (0.0) is used.

    The CSV file will have two columns: 'Date' and 'Sentiment_Mean'.

    Example:
    If the data contains:
    {
        '2023-08-11': {'processed_values': [...], 'arithmetic_mean': 0.75},
        '2023-08-13': {'processed_values': [...], 'arithmetic_mean': 0.60}
    }
    The generated CSV will have:
    Date,Sentiment_Mean
    2023-08-11,0.75
    2023-08-12,0.0
    2023-08-13,0.60

    Note:
    Make sure the 'datetime' module is imported at the beginning of the script.

    """
    # Get a list of all dates in the data
    all_dates = [datetime.strptime(date, '%Y-%m-%d') for date in data.keys()]

    # Find the minimum and maximum dates
    min_date = min(all_dates)
    max_date = max(all_dates)

    # Create a list of dates in the date range
    date_range = [min_date + timedelta(days=i)
                  for i in range((max_date - min_date).days + 1)]

    with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
        fieldnames = ['Date', 'Sentiment_Mean']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for date in date_range:
            formatted_date = date.strftime('%Y-%m-%d')
            if formatted_date in data:
                writer.writerow({
                    'Date': formatted_date,
                    'Sentiment_Mean': data[formatted_date]['arithmetic_mean']
                })
            else:
                writer.writerow({
                    'Date': formatted_date,
                    'Sentiment_Mean': 0.0  # Or any default value you prefer
                })


def main():
    """
    Main function to process sentiment data.

    Reads raw sentiment data from a JSON file, applies multipliers, calculates mean,
    saves the processed data to a JSON file, and writes sentiment means to a CSV file.
    """
    # Load raw data
    data = load_data(INPUT_FILE)

    # Process the data
    processed_data = process_data(data, MULTIPLIERS)

    # Save the updated data
    save_data(processed_data, OUTPUT_FILE)

    # Write sentiment means to CSV
    write_csv(processed_data)

    print("Processed Sentiment Data Successfully!")


if __name__ == "__main__":
    main()
