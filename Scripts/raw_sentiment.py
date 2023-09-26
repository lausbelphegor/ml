"""
This script performs sentiment analysis on news data and saves the results to a JSON file.

It uses the Transformers library's pre-trained 'ProsusAI/finbert' model to analyze the sentiment of news titles.
Sentiment labels are categorized as negative, neutral, or positive, with corresponding numerical values -1, 0, and 1.
The sentiment scores and labels for each news title are stored in a JSON file organized by date.

Requirements:
- transformers library (for the sentiment analysis model)
- datetime library (for date manipulation)
- The 'bitcoin_news_data.json' file containing a list of news entries with "title" and "date" fields.

Output:
- 'raw_sentiment.json': A JSON file containing sentiment scores and labels grouped by date.

Author: og
Date: 2023-08-11
"""
import configparser
import json
from datetime import datetime
from transformers import pipeline
import helper_functions as hf
from tqdm import tqdm

hf.change_working_directory()


SLV = {
    "NEG": {"LABEL": 'negative', "VALUE": -1},
    "NEUT": {"LABEL": 'neutral', "VALUE": 0},
    "POS": {"LABEL": 'positive', "VALUE": 1}
}

config = configparser.ConfigParser()
config.read('config.ini')

TEST_MODE = config['Settings']['testmode'] == 'True'

DATA_FILE = config['Paths']['inputsentiment'] if not TEST_MODE else config['Paths']['TEST_inputsentiment']
OUTPUT_FILE = config['Paths']['outputsentiment'] if not TEST_MODE else config['Paths']['TEST_outputsentiment']

SENTIMENT_PIPELINE = "ProsusAI/finbert"


def load_json_data(file_path):
    """
    Load JSON data from the specified file path.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        list: Loaded JSON data.
    """
    with open(file_path, 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def preprocess_date(date_str):
    """
    Convert a date string to a formatted date string (YYYY-MM-DD).

    Parameters:
        date_str (str): Date string in the format "Mon DD, YYYY".

    Returns:
        str: Formatted date string in "YYYY-MM-DD" format.
    """
    date = datetime.strptime(date_str, "%b %d, %Y").date()
    formatted_date = date.strftime("%Y-%m-%d")
    return formatted_date


def analyze_sentiment(title, sentiment_pipe):
    """
    Analyze the sentiment of the given title using the sentiment pipeline.

    Parameters:
        title (str): The news article title for sentiment analysis.
        sentiment_pipe: The sentiment analysis pipeline instance.

    Returns:
        list: Sentiment label and score in the format [label, score].
    """
    raw_sentiment = sentiment_pipe(title)
    sentiment = [0, 0]  # [label, score]

    sentiment[0] = SLV["NEG"]["VALUE"] if (raw_sentiment[0]["label"] == SLV["NEG"]["LABEL"]) else (
        SLV["POS"]["VALUE"] if (raw_sentiment[0]["label"] == SLV["POS"]["LABEL"]) else SLV["NEUT"]["VALUE"])

    sentiment[1] = raw_sentiment[0]["score"]
    return sentiment


def process_data(data, sentiment_pipe):
    """
    Process news data and analyze sentiment for each entry.

    Parameters:
        data (list): List of news entries with "title" and "date" fields.
        sentiment_pipe: The sentiment analysis pipeline instance.

    Returns:
        dict: Sentiment scores grouped by date.
    """
    sentiment_scores = {}  # Dictionary to store sentiment scores by date

    # Use tqdm to display a progress bar
    for entry in tqdm(data, desc="Analyzing Sentiment", unit="entry"):
        title = entry["title"]
        date_str = entry["date"]
        formatted_date = preprocess_date(date_str)

        sentiment = analyze_sentiment(title, sentiment_pipe)

        if formatted_date in sentiment_scores:
            sentiment_scores[formatted_date].append(sentiment)
        else:
            sentiment_scores[formatted_date] = [sentiment]

    return sentiment_scores


def save_to_json(sentiment_scores, output_file):
    """
    Save sentiment scores to a JSON file.

    Parameters:
        sentiment_scores (dict): Sentiment scores grouped by date.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding="utf-8") as json_file:
        json.dump(sentiment_scores, json_file, indent=4)


def main():
    """
    Perform sentiment analysis on news data and save the results to a JSON file.

    This function orchestrates the sentiment analysis process by loading news data,
    applying sentiment analysis using a predefined pipeline, and then saving the sentiment
    scores by date to a JSON file.

    Steps:
    1. Load news data from the specified JSON file.
    2. If in testing mode (TEST_MODE=True), limit the data to a subset for testing purposes.
    3. Initialize a sentiment analysis pipeline using the configured model.
    4. Process each entry in the data, analyze sentiment, and group scores by date.
    5. Save the sentiment scores to a JSON file.

    Configuration:
    - DATA_FILE: Path to the JSON file containing news data.
    - OUTPUT_FILE: Path to the JSON file to save sentiment scores.
    - NUM_ENTRIES_TO_PROCESS: Number of news entries to process for testing.
    - SENTIMENT_PIPELINE: Predefined pipeline model for sentiment analysis.
    - TEST_MODE: Set to True for processing a limited subset of data for testing.

    Note:
    The sentiment analysis pipeline should be preconfigured using the 'transformers' library.
    """
    data = load_json_data(DATA_FILE)

    if TEST_MODE:
        # Limit data for testing
        data = data[:int(config['Settings']
                         ['sentiment_analysis_test_entries'])]

    sentiment_pipe = pipeline("text-classification", model=SENTIMENT_PIPELINE)

    sentiment_scores = process_data(data, sentiment_pipe)

    save_to_json(sentiment_scores, OUTPUT_FILE)

    print("Analyzed News Titles Successfully!")


if __name__ == "__main__":
    main()
