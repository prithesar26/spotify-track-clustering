# Spotify Track Clustering

## Objective
Group Spotify songs by audio characteristics using unsupervised learning.

## Dataset
Source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

The original dataset contains 114,000 rows and 21 columns.
Removing repeated track IDs leaves 89,741 unique tracks.

## Features
Danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, and tempo.

No target variable is used. Popularity and track_genre are excluded
from clustering.

## Workflow
1. Explore missing values, duplicates, distributions, and correlations.
2. Remove repeated track IDs and fill missing song details.
3. Apply median imputation and StandardScaler to numeric features.
4. Compare K-Means and DBSCAN on a reproducible 10,000-song sample.
5. Test K-Means with 2–12 clusters.
6. Evaluate silhouette scores, cluster sizes, and DBSCAN noise.
7. Visualize clusters using PCA and interpret feature averages.
8. Fit the selected K-Means model on all unique tracks.
9. Save the preprocessing pipeline and models.
10. Build a Streamlit application.

## Model Comparison
| Algorithm | Clusters | Sampled silhouette | Noise |
|-----------|----------|--------------------|-------|
| K-Means | 2 | 0.2531 | 0% |
| DBSCAN | 2 | 0.5827 | 6.75% |

DBSCAN silhouette excludes noise points, so the scores cover
different subsets of songs.

DBSCAN produced one large cluster of 9,263 songs and one small
cluster of 62 songs. K-Means was selected because its groups were
more useful for organizing all songs and it supports new inputs.

## Final Results
Final K-Means sampled silhouette score: 0.2594.

- Acoustic and Low Energy: 22,281 songs.
- Energetic and Danceable: 67,460 songs.

The groups overlap and represent broad audio characteristics.
Silhouette score is not prediction accuracy.

## Application
- Browse songs by cluster.
- Search by song name or artist.
- Enter nine audio features to find a cluster.
- View example songs from that cluster.

Examples are not ranked recommendations.

## Run Locally
Install dependencies:
`python -m pip install -r requirements.txt`

Start the app:
`python -m streamlit run app.py`

Keep spotify_clustering.joblib and spotify_clustered.csv
in the same folder as app.py.

## Notebook
Open spotify.ipynb with dataset.csv in the same folder.
The notebook also requires matplotlib, seaborn, and Jupyter.

## Deployment
Public application URL: To be added after deployment.
