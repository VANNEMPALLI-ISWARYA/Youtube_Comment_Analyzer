# youtube_comments.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

# Your YouTube API key here (replace 'YOUR_API_KEY' with your actual key)
API_KEY = 'AIzaSyAdpcJhleqK2psoUodMBsThdX_xqX8NYCw'

# YouTube API client setup
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_id(url):
    # Extract video ID from YouTube URL
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_comments(video_id, max_comments=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= max_comments:
                return comments
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response['nextPageToken'],
                textFormat="plainText"
            )
            response = request.execute()
        else:
            break
    return comments

def clean_text(text):
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def analyze_sentiment(comment):
    # Use TextBlob to get sentiment polarity
    return TextBlob(comment).sentiment.polarity

def main():
    video_url = input("Enter YouTube Video URL: ")
    video_id = get_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL.")
        return

    print("Fetching comments...")
    comments = get_comments(video_id, max_comments=200)

    if not comments:
        print("No comments found.")
        return

    # Save comments in DataFrame
    df = pd.DataFrame(comments, columns=['comment'])
    df['cleaned'] = df['comment'].apply(clean_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['filtered'] = df['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Sentiment analysis
    df['sentiment'] = df['filtered'].apply(analyze_sentiment)

    # Classify sentiment
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    # Show sentiment counts
    print("\nSentiment counts:")
    print(df['sentiment_label'].value_counts())

    # Plot sentiment pie chart
    plt.figure(figsize=(6,6))
    df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'red', 'gray'])
    plt.title('Comment Sentiment Distribution')
    plt.ylabel('')
    plt.show()

    # Combine all filtered comments for word cloud
    text_combined = ' '.join(df['filtered'])

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Comments')
    plt.show()

    # Top 10 words frequency bar chart
    all_words = ' '.join(df['filtered']).split()
    word_freq = pd.Series(all_words).value_counts().head(10)

    plt.figure(figsize=(10,5))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette='viridis')
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.show()

if __name__ == '__main__':
    main()
