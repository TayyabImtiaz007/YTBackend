from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from googleapiclient.discovery import build
from collections import Counter
import matplotlib.pyplot as plt
import base64
import io
import os
from dotenv import load_dotenv

app = Flask(__name__)


load_dotenv()  # This loads the variables from .env into the environment
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
 
# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    sentiment = "positive" if probabilities[0][1] > probabilities[0][0] else "negative"
    return sentiment

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_video_stats(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(part="statistics,snippet", id=video_id)
    response = request.execute()
    
    if 'items' in response:
        item = response['items'][0]
        stats = item['statistics']
        snippet = item['snippet']
        return {
            'title': snippet['title'],
            'views': stats['viewCount'],
            'likes': stats.get('likeCount', 'N/A'),
            'comments': stats['commentCount']
        }
    return None

def get_video_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    
    while request and len(comments) < max_results:
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        request = youtube.commentThreads().list_next(request, response)
    
    return comments[:max_results]

def create_sentiment_graph(sentiments):
    counts = Counter(sentiments)
    
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values())
    plt.title('Sentiment Analysis of Video Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form['video_url']
    
    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'})
    
    video_stats = get_video_stats(video_id)
    if not video_stats:
        return jsonify({'error': 'Could not fetch video statistics'})
    
    comments = get_video_comments(video_id)
    sentiments = [analyze_sentiment(comment) for comment in comments]
    
    sentiment_counts = Counter(sentiments)
    overall_sentiment = "positive" if sentiment_counts['positive'] > sentiment_counts['negative'] else "negative"
    
    graph = create_sentiment_graph(sentiments)
    
    return jsonify({
        'video_stats': video_stats,
        'sentiment_counts': dict(sentiment_counts),
        'overall_sentiment': overall_sentiment,
        'graph': graph
    })

