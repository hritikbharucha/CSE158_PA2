import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import precision_score, recall_score

# Function to load data from JSON files in the main data folder
def load_playlists(data_path, num_files=5):
    playlists = []
    files = os.listdir(data_path)[:num_files]  # Limit to num_files for analysis
    for file in files:
        file_path = os.path.join(data_path, file)
        if file.endswith('.json'):  # Ensure we're processing only JSON files
            with open(file_path, 'r') as f:
                data = json.load(f)
                playlists.extend(data['playlists'])
    return playlists

# Preprocessing to extract relevant data
def preprocess_playlists(playlists):
    all_tracks = []
    track_popularity = Counter()
    
    for playlist in playlists:
        for track in playlist['tracks']:
            track_id = track['track_uri']
            all_tracks.append(track_id)
            track_popularity[track_id] += 1
    
    return all_tracks, track_popularity

# Baseline: Popularity-based recommender
def recommend_popularity(track_popularity, known_tracks, k=10):
    """Recommend top k most popular tracks not in known_tracks."""
    most_popular = [
        track for track, _ in track_popularity.most_common()
        if track not in known_tracks
    ]
    return most_popular[:k]

# Evaluation Metrics
def precision_at_k(actual, predicted, k):
    predicted = set(predicted[:k])
    actual = set(actual)
    return len(predicted & actual) / k

def recall_at_k(actual, predicted, k):
    predicted = set(predicted[:k])
    actual = set(actual)
    return len(predicted & actual) / len(actual) if actual else 0

# Main Execution
def main(data_path, num_files=5, k=10):
    # Load and preprocess data
    playlists = load_playlists(data_path, num_files=num_files)
    all_tracks, track_popularity = preprocess_playlists(playlists)
    
    # Split data into train and test
    train_playlists, test_playlists = train_test_split(playlists, test_size=0.2, random_state=42)
    
    # Evaluate on test playlists
    precision_scores = []
    recall_scores = []
    
    for playlist in test_playlists:
        if len(playlist['tracks']) < 5:
            continue  # Skip short playlists
        
        known_tracks = [track['track_uri'] for track in playlist['tracks'][:-2]]
        actual_tracks = [track['track_uri'] for track in playlist['tracks'][-2:]]
        
        # Recommend using the baseline popularity model
        recommendations = recommend_popularity(track_popularity, known_tracks, k=k)
        
        # Compute metrics
        precision = precision_at_k(actual_tracks, recommendations, k)
        recall = recall_at_k(actual_tracks, recommendations, k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Print results
    print(f"Average Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Average Recall@{k}: {np.mean(recall_scores):.4f}")

if __name__ == "__main__":
    # Change this path to your dataset folder
    data_path = "data"
    main(data_path, num_files=15, k=10)
