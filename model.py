import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

class SpotifyRecommender:
    def __init__(self, data_path: str, num_files: int = 50):
        """
        Initialize recommender with enhanced data loading and preprocessing
        
        Args:
            data_path (str): Path to playlist JSON files
            num_files (int): Number of files to load
        """
        self.playlists = self.load_data(data_path, num_files)
        self.track_map = {}
        self.interaction_matrix = None
        self.song_name_matrix = None
        self.song_map = {}
        
        # Preprocessing steps
        self.preprocess_data()
    
    def load_data(self, data_path: str, num_files: int) -> List[Dict]:
        """
        Enhanced data loading with error handling and logging
        
        Args:
            data_path (str): Directory containing playlist JSON files
            num_files (int): Maximum number of files to load
        
        Returns:
            List of playlist dictionaries
        """
        playlists = []
        files = [f for f in os.listdir(data_path) if f.endswith('.json')][:num_files]
        
        for file in files:
            try:
                with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    playlists.extend(data.get('playlists', []))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading file {file}: {e}")
        
        print(f"Loaded {len(playlists)} playlists from {len(files)} files")
        return playlists
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing pipeline
        """
        # Build interaction matrix with weighted interactions
        self.interaction_matrix, self.track_map = self.build_interaction_matrix()
        
        # Vectorize song names with enhanced features
        self.song_name_matrix, self.song_map, _ = self.vectorize_song_names()
        print("Data preprocessing complete")
    
    def build_interaction_matrix(self) -> Tuple[np.ndarray, Dict]:
        """
        Build an enhanced interaction matrix with weighted interactions
        
        Returns:
            Numpy interaction matrix and track mapping
        """
        print("Building interaction matrix...")
        # Extract unique tracks with frequency
        track_freq = {}
        for playlist in self.playlists:
            for track in playlist.get('tracks', []):
                track_uri = track.get('track_uri')
                track_freq[track_uri] = track_freq.get(track_uri, 0) + 1
        
        # Create track map and matrix
        track_ids = sorted(track_freq.keys(), key=lambda x: track_freq[x], reverse=True)
        track_map = {track: i for i, track in enumerate(track_ids)}

        print(f"Found {len(track_ids)} unique tracks")
        
        playlist_matrix = []
        for playlist in self.playlists:
            row = np.zeros(len(track_ids))
            for track in playlist.get('tracks', []):
                track_uri = track.get('track_uri')
                # Weighted by track frequency and potentially playlist popularity
                row[track_map[track_uri]] = track_freq[track_uri]
            playlist_matrix.append(row)
        
        print(f"Built interaction matrix with {len(track_ids)} unique tracks")
        return np.array(playlist_matrix), track_map
    
    def vectorize_song_names(self) -> Tuple[np.ndarray, Dict, TfidfVectorizer]:
        """
        Enhanced song name vectorization
        
        Returns:
            TF-IDF matrix, song map, and vectorizer
        """
        song_names = []
        song_map = {}
        
        for playlist in self.playlists:
            for track in playlist.get('tracks', []):
                track_uri = track.get('track_uri')
                track_name = track.get('track_name', '')
                if track_uri not in song_map and track_name:
                    song_names.append(track_name)
                    song_map[track_uri] = track_name
        
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english', 
            ngram_range=(1, 2)  # Consider bigrams for better context
        )
        song_name_matrix = vectorizer.fit_transform(song_names)
        
        print(f"Vectorized {len(song_map)} unique song names")
        return song_name_matrix, song_map, vectorizer
    
    def content_based_recommendation(self, k: int = 10) -> List[List[str]]:
        """
        Enhanced content-based recommendation
        
        Args:
            k (int): Number of recommendations per playlist
        
        Returns:
            List of recommended track URIs
        """
        playlist_names = [p.get('name', '') for p in self.playlists]
        playlist_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        playlist_vectors = playlist_vectorizer.fit_transform(playlist_names)
        
        content_recommendations = []
        for playlist_vector in playlist_vectors:
            # Compute content similarity
            similarities = cosine_similarity(playlist_vector, self.song_name_matrix)
            top_indices = np.argsort(-similarities[0])[:k]
            
            # Get top song URIs
            top_songs = [list(self.song_map.keys())[idx] for idx in top_indices]
            content_recommendations.append(top_songs)
        
        print(f"Generated content-based recommendations for {len(content_recommendations)} playlists")
        return content_recommendations
    
    def collaborative_filtering(self, k: int = 10) -> np.ndarray:
        """
        Advanced collaborative filtering using SVD
        
        Args:
            k (int): Number of components for dimensionality reduction
        
        Returns:
            Reduced dimension matrix
        """
        svd = TruncatedSVD(n_components=min(k, self.interaction_matrix.shape[1]-1), random_state=42)
        print(f"Performing collaborative filtering with {svd.n_components} components")
        return svd.fit_transform(self.interaction_matrix)
    
    def hybrid_recommendation(self, alpha: float = 0.5, k: int = 10) -> List[List[str]]:
        """
        Advanced hybrid recommendation system
        
        Args:
            alpha (float): Weight for collaborative filtering
            k (int): Number of recommendations
        
        Returns:
            Hybrid recommendations
        """
        # Collaborative filtering
        cf_matrix = self.collaborative_filtering(k)
        
        # Content-based recommendations
        cb_recommendations = self.content_based_recommendation(k)
        
        # Combine recommendations
        hybrid_recommendations = []
        for i in range(len(self.playlists)):
            # Combine CF and content-based scores
            cf_scores = cf_matrix[i]
            cb_tracks = [self.track_map.get(uri, -1) for uri in cb_recommendations[i]]
            
            # Filter out invalid track indices
            valid_cb_tracks = [idx for idx in cb_tracks if idx != -1]
            
            # Hybrid scoring
            combined_scores = {}
            for idx in range(len(self.track_map)):
                cf_score = cf_scores[idx] if idx < len(cf_scores) else 0
                cb_score = 1 if idx in valid_cb_tracks else 0
                combined_scores[idx] = alpha * cf_score + (1 - alpha) * cb_score
            
            # Get top k recommendations
            top_tracks = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
            hybrid_recommendations.append([list(self.track_map.keys())[track] for track in top_tracks])
        
        print(f"Generated hybrid recommendations for {len(hybrid_recommendations)} playlists")
        return hybrid_recommendations
    
    def evaluate_recommendations(self, test_playlists: List[Dict], predictions: List[List[str]], k: int = 10):
        """
        Enhanced recommendation evaluation
        
        Args:
            test_playlists (List[Dict]): Test playlist data
            predictions (List[List[str]]): Recommended track URIs
            k (int): Top-k recommendations to evaluate
        """
        precision_scores = []
        recall_scores = []
        
        for i, playlist in enumerate(test_playlists):
            actual_tracks = set(track['track_uri'] for track in playlist.get('tracks', []))
            predicted_tracks = set(predictions[i][:k])
            
            # Handle edge cases
            if not actual_tracks:
                continue
            
            precision = len(actual_tracks & predicted_tracks) / min(k, len(actual_tracks))
            recall = len(actual_tracks & predicted_tracks) / len(actual_tracks)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
        print(f"Recall@{k}: {np.mean(recall_scores):.4f}")

def main(data_path: str, num_files: int = 50, k: int = 10, alpha: float = 0.5):
    """
    Main execution function for Spotify playlist recommender
    
    Args:
        data_path (str): Directory with playlist JSON files
        num_files (int): Number of files to process
        k (int): Number of recommendations
        alpha (float): Hybrid recommendation weight
    """
    # Create recommender
    recommender = SpotifyRecommender(data_path, num_files)
    
    # Split data for training and testing
    train_playlists, test_playlists = train_test_split(recommender.playlists, test_size=0.2, random_state=42)
    
    # Generate hybrid recommendations
    predictions = recommender.hybrid_recommendation(alpha, k)
    
    # Evaluate recommendations
    recommender.evaluate_recommendations(test_playlists, predictions, k)

if __name__ == "__main__":
    main(data_path="data", num_files=15, k=10, alpha=0.5)