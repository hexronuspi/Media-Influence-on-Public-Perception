#to be run in google colab
!pip install snscrape
import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(#Russia) until:2023-06-26 since:2023-06-22"


tweets = []
limit = 1000

for tweet in sntwitter.TwitterHashtagScraper(query).get_items():
    if len(tweets) == limit:
        break
    tweets.append([
        tweet.user.displayname,
        tweet.url,
        tweet.user.username,
        tweet.sourceLabel,
        tweet.user.location,
        tweet.content
    ])

df = pd.DataFrame(tweets, columns=['Name', 'URL', 'Username', 'Source', 'Location', 'Content'])
df.to_csv('tweets.csv', index=False)
print("Data collection completed and saved to tweets.csv file.")

import pandas as pd

import numpy as np

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid=SentimentIntensityAnalyzer()

# Read the CSV file using Pandas
df = pd.read_csv('tweets.csv')

print(df.head())

df.info()

df['Content'] = df['Content'].replace(np.nan, '', regex=True)

sentiment_scores = []
for text in df['Content']:
  scores = sid.polarity_scores(text)
  sentiment_scores.append(scores['compound'])

df['Sentiment Score'] = sentiment_scores

def get_sentiment_lable(score):
    if score>=0.05:
      return 'Positive'
    elif score<=-0.05:
      return 'Negative'
    else:
      return 'Neutral'

sentiment_lables = df['Sentiment Score'].apply(get_sentiment_lable)

df['Sentiment Lable'] = sentiment_lables

df.head()

import matplotlib.pyplot as plt

Sentiment_counts = df['Sentiment Lable'].value_counts()

plt.pie(Sentiment_counts, labels = Sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.show()
