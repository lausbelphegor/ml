"""
Module: Article Crawler

This module contains functions to crawl a website, extract information about articles related to Bitcoin,
and organize the extracted data.

Dependencies:
- time
- selenium
- webdriver (Chrome)


Author: og
Date: 2023-08-11
"""
import configparser
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import helper_functions as hf

hf.change_working_directory()

config = configparser.ConfigParser()
config.read('config.ini')


def extract_info_from_block(block):
    """
    Extracts information from a single article block.

    Args:
        block (WebElement): The WebElement representing the article block.

    Returns:
        dict: A dictionary containing article information.
    """
    title_element = block.find_element(By.CLASS_NAME, 'sc-cdDgOI')
    title = title_element.text
    link = title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')

    category_element = block.find_element(By.CLASS_NAME, 'sc-jLHygt')
    category = category_element.text

    date_element = block.find_element(By.TAG_NAME, 'time')
    date = date_element.get_attribute('datetime')

    description_element = block.find_element(By.CLASS_NAME, 'sc-ggIBej')
    description = description_element.text

    return {
        'title': title,
        'link': link,
        'category': category,
        'date': date,
        'description': description
    }


def extract_info_from_page(driver, url):
    """
    Extracts information from articles on a given page.

    Args:
        driver (WebDriver): The Selenium WebDriver instance.
        url (str): The URL of the page to extract information from.

    Returns:
        list: A list of dictionaries containing article information.
    """
    driver.get(url)

    # Wait for the article blocks to be present on the page
    WebDriverWait(driver, int(config['Settings']['crawler_timeout'])).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'sc-cemHvA')))

    articles = []
    article_blocks = driver.find_elements(By.CLASS_NAME, 'sc-cemHvA')

    # Extract information from each article block
    for block in article_blocks:
        article_info = extract_info_from_block(block)
        articles.append(article_info)

    return articles


def crawl_and_extract_pages(base_url, num_pages):
    """
    Crawls through multiple pages and extracts article information.

    Args:
        base_url (str): The base URL of the website.
        num_pages (int): The number of pages to crawl.

    Returns:
        list: A list of dictionaries containing article information.
    """
    # Make sure to provide the path to the ChromeDriver executable
    options = webdriver.ChromeOptions()

    # Set the window size to maximize
    options.add_argument("start-maximized")

    # Disable the "Chrome is being controlled by automated test software" notification
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    # Initialize a Selenium webdriver
    driver = webdriver.Chrome(options)

    articles = []

    # Crawl through each page and extract article information
    for page_num in range(1, num_pages + 1):
        page_url = f"{base_url}/page/{page_num}?s=bitcoin"
        page_articles = extract_info_from_page(driver, page_url)
        articles.extend(page_articles)
        print(f"Extracted {len(page_articles)} articles from page {page_num}")

    driver.quit()
    return articles


def crawl_articles(url, page):
    """
    Crawl and extract articles from a news website for a given page.

    Args:
        url (str): URL of the news website.
        page (int): Page number to crawl.

    Returns:
        list: List of extracted articles.
    """
    return crawl_and_extract_pages(url, page)


def save_articles_to_json(name, articles):
    """
    Save extracted articles to a JSON file.

    Args:
        name (str): Name of the cryptocurrency.
        articles (list): List of articles to be saved.
    """
    json_file_path = f'{name}'
    with open(json_file_path, 'w', encoding="utf-8") as json_file:
        json.dump(articles, json_file, indent=4)


def crawl_and_save_all_articles(name, url, num_pages):
    """
    Crawl and extract all articles from a news website, and save the data to a JSON file.

    Args:
        name (str): Name of the cryptocurrency.
        url (str): URL of the news website.

    Returns:
        list: List of extracted articles.
    """
    all_articles = []
    all_articles = crawl_articles(url, num_pages)

    save_articles_to_json(name, all_articles)

    return all_articles
