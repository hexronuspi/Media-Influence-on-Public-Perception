# Twitter Sentiment Analysis with snscrape

## Overview
This repository offers a machine learning model that gathers tweets for sentiment analysis using snscrape, a Python package for scraping data from Twitter. The methodology attempts to categorise tweet sentiment as good, negative, or neutral, so offering insights into public thoughts and attitudes.

## Features
- **Data Scraping**: The model leverages snscrape to scrape tweets based on user-defined search queries, hashtags, or user profiles. The scraped data includes the tweet text, user information, timestamp, and other relevant metadata.
- **Sentiment Analysis**: Using natural language processing (NLP) techniques, the model analyzes the scraped tweets and assigns sentiment labels (positive, negative, or neutral) to each tweet. This allows for quantifying the overall sentiment of a specific topic or assessing the sentiment of individual tweets.

## Installation
To use this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/hexronuspi/Model-Test-twitter`
2. Run the python code:  Model-Test-twitter.py  -> This code was tested to run in google colab, to run it on your local system, necessary changes has to be done.

## Usage
1. Access the web interface.
2. Enter your search query or hashtag to scrape relevant tweets from Twitter.
3. Wait for the scraping process to complete.
4. Once the scraping is finished, the sentiment analysis will be performed on the scraped tweets.
5. Explore the sentiment analysis results, including sentiment distribution visualizations.

## Contributing
Contributions to this repository are welcome! If you want to contribute, please follow these steps:

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments
- The [snscrape](https://github.com/JustAnotherArchivist/snscrape) library for providing an easy-to-use Twitter scraping functionality.
- The contributors and maintainers of the Python libraries used in this project, including scikit-learn, NLTK, Flask, and more.
- ` @hexronuspi ` and future contributors.

Please feel free to provide feedback or suggestions
