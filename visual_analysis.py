import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
data_path = "data"

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

# Load data
num_files = 15  # Modify based on how much data you want to analyze
playlists = load_playlists(data_path, num_files=num_files)

# Convert data into a DataFrame for analysis
playlists_df = pd.json_normalize(playlists)

# Basic Statistics
print("Number of playlists:", playlists_df.shape[0])
print(playlists_df.describe())

# Visualizing Playlist Statistics
def plot_basic_stats(playlists_df):
    # Histogram of the number of tracks per playlist
    plt.figure(figsize=(10, 6))
    sns.histplot(playlists_df['num_tracks'], bins=30, kde=False, color='blue')
    plt.title("Distribution of Number of Tracks in Playlists")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Frequency")
    plt.show()

    # Number of followers per playlist
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=playlists_df['num_followers'])
    plt.title("Number of Followers per Playlist")
    plt.xlabel("Followers")
    plt.show()

# Call visualization function
plot_basic_stats(playlists_df)

# Analyze Track Data
tracks_data = []
for playlist in playlists:
    for track in playlist['tracks']:
        tracks_data.append(track)

tracks_df = pd.DataFrame(tracks_data)

# Most Popular Artists
top_artists = tracks_df['artist_name'].value_counts().head(10)

# Plot Top Artists
plt.figure(figsize=(10, 6))
sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
plt.title("Top 10 Artists in the Dataset")
plt.xlabel("Number of Tracks")
plt.ylabel("Artist Name")
plt.show()

# Analyze Average Playlist Length by Number of Tracks
playlists_df['duration_minutes'] = playlists_df['duration_ms'] / 60000

plt.figure(figsize=(10, 6))
sns.scatterplot(x=playlists_df['num_tracks'], y=playlists_df['duration_minutes'], alpha=0.6)
plt.title("Playlist Length vs. Number of Tracks")
plt.xlabel("Number of Tracks")
plt.ylabel("Playlist Length (minutes)")
plt.show()

# Analyze Collaborative Playlists
collaborative_counts = playlists_df['collaborative'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=collaborative_counts.index, y=collaborative_counts.values, palette="muted")
plt.title("Collaborative vs. Non-Collaborative Playlists")
plt.xlabel("Collaborative")
plt.ylabel("Number of Playlists")
plt.xticks([0, 1], ['Non-Collaborative', 'Collaborative'])
plt.show()

# Additional Insights and Visualizations

# 1. Analyze the distribution of playlists by theme (e.g., playlist names)
def analyze_playlist_names(playlists_df):
    # Extract most common words in playlist names
    playlist_names = playlists_df['name'].dropna().str.lower().str.split()
    all_words = [word for name in playlist_names for word in name]
    word_freq = pd.Series(all_words).value_counts().head(20)

    # Plot Top Words
    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette="coolwarm")
    plt.title("Top 20 Most Common Words in Playlist Names")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()

analyze_playlist_names(playlists_df)

# 2. Track Popularity Distribution
def track_popularity_distribution(tracks_df):
    # Histogram of track frequency in playlists
    track_counts = tracks_df['track_name'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(track_counts, bins=50, kde=False, color='purple', log_scale=True)
    plt.title("Distribution of Track Popularity in Playlists")
    plt.xlabel("Number of Playlists")
    plt.ylabel("Frequency")
    plt.show()

track_popularity_distribution(tracks_df)

# 3. Artist Diversity in Playlists
def artist_diversity(playlists):
    artist_counts = []
    for playlist in playlists:
        unique_artists = {track['artist_name'] for track in playlist['tracks']}
        artist_counts.append(len(unique_artists))

    plt.figure(figsize=(10, 6))
    sns.histplot(artist_counts, bins=30, kde=False, color='green')
    plt.title("Distribution of Artist Diversity in Playlists")
    plt.xlabel("Number of Unique Artists per Playlist")
    plt.ylabel("Frequency")
    plt.show()

artist_diversity(playlists)

# 4. Playlist Length and Diversity Correlation
def length_vs_diversity(playlists):
    playlist_stats = []
    for playlist in playlists:
        num_tracks = len(playlist['tracks'])
        unique_artists = len({track['artist_name'] for track in playlist['tracks']})
        playlist_stats.append((num_tracks, unique_artists))

    stats_df = pd.DataFrame(playlist_stats, columns=['num_tracks', 'unique_artists'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stats_df, x='num_tracks', y='unique_artists', alpha=0.6)
    plt.title("Playlist Length vs. Artist Diversity")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Number of Unique Artists")
    plt.show()

length_vs_diversity(playlists)

# 5. Temporal Analysis (if timestamp is available)
def temporal_analysis(playlists_df):
    if 'modified_at' in playlists_df.columns:
        # Convert timestamps to datetime
        playlists_df['modified_at'] = pd.to_datetime(playlists_df['modified_at'], unit='s')
        playlists_df['year'] = playlists_df['modified_at'].dt.year

        # Plot number of playlists over time
        playlists_per_year = playlists_df['year'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=playlists_per_year.index, y=playlists_per_year.values, marker='o')
        plt.title("Number of Playlists Created/Modified Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Playlists")
        plt.show()

temporal_analysis(playlists_df)


