"""
Cryptocurrency News Crawler Script

This script crawls a cryptocurrency news website, extracts articles, and saves the data to a JSON file.

Author: og
Date: 2023-08-11
"""
import configparser
import bitcoin_crwler_functions as bcf
import helper_functions as hf

hf.change_working_directory()

config = configparser.ConfigParser()
config.read('config.ini')

TEST_MODE = config['Settings']['testmode'] == 'True'

OUTPUT_FILE = config['Paths']['outputcrawler'] if not TEST_MODE else config['Paths']['TEST_outputcrawler']
NEWS_URL = config['URLs']['news']

NUM_PAGES = 2666 if not TEST_MODE else int(
    config['Settings']['crawler_test_pages'])


def main(output_file, url, num_pages):
    """
    Main function to crawl news and save data.

    This function initializes parameters using constants and calls the 'crawl_and_save_all_articles()'
    function to crawl the website and save the extracted data. Finally, it prints the number of
    extracted articles and the path to the saved JSON file.
    """
    all_articles = bcf.crawl_and_save_all_articles(output_file, url, num_pages)

    print(
        f"Extracted {len(all_articles)} articles and saved to '{output_file}.json'.")

    print("Crawled and Extracted Articles Successfully!")


if __name__ == "__main__":
    main(OUTPUT_FILE, NEWS_URL, NUM_PAGES)
