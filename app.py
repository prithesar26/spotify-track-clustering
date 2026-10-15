from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spotify Track Clustering",
    page_icon="🎵",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "spotify_clustering.joblib")


@st.cache_data
def load_songs():
    return pd.read_csv(BASE_DIR / "spotify_clustered.csv")


bundle = load_model()
songs = load_songs()

pipeline = bundle["pipeline"]
features = bundle["features"]
cluster_names = bundle["cluster_names"]

st.title("🎵 Spotify Track Clustering")
st.write(
    "Explore songs grouped by their audio characteristics, "
    "or enter audio features to find a cluster."
)

st.caption(
    "K-Means with 2 clusters | "
    f"Sampled silhouette score: "
    f"{bundle['sampled_silhouette_score']:.4f}"
)

st.info(
    "These are broad audio groups with some overlap. "
    "The app groups songs; it does not predict popularity or genre."
)

explore_tab, predict_tab = st.tabs([
    "Explore Songs",
    "Find a Cluster"
])

with explore_tab:
    selected_cluster = st.selectbox(
        "Choose a song group",
        options=sorted(cluster_names),
        format_func=lambda value: cluster_names[value]
    )

    group = songs[
        songs["cluster"] == selected_cluster
    ].copy()

    st.metric("Songs in this group", f"{len(group):,}")

    search = st.text_input("Search by song or artist")

    if search.strip():
        query = search.strip()

        matches = (
            group["track_name"].fillna("").str.contains(
                query, case=False, regex=False
            )
            | group["artists"].fillna("").str.contains(
                query, case=False, regex=False
            )
        )

        group = group[matches]

    st.write(f"Matching songs: {len(group):,}")

    st.dataframe(
        group[
            ["track_name", "artists", "cluster_name"]
        ].head(100),
        hide_index=True,
        use_container_width=True
    )

    st.caption("Showing up to 100 matching songs.")

with predict_tab:
    st.write(
        "Enter original audio feature values. "
        "The saved pipeline handles scaling automatically."
    )

    with st.form("audio_features"):
        left, right = st.columns(2)

        with left:
            danceability = st.slider(
                "Danceability", 0.0, 1.0, 0.5, 0.01
            )
            energy = st.slider(
                "Energy", 0.0, 1.0, 0.5, 0.01
            )
            loudness = st.slider(
                "Loudness (dB)", -60.0, 5.0, -8.0, 0.1
            )
            speechiness = st.slider(
                "Speechiness", 0.0, 1.0, 0.05, 0.01
            )
            acousticness = st.slider(
                "Acousticness", 0.0, 1.0, 0.3, 0.01
            )

        with right:
            instrumentalness = st.slider(
                "Instrumentalness", 0.0, 1.0, 0.0, 0.01
            )
            liveness = st.slider(
                "Liveness", 0.0, 1.0, 0.2, 0.01
            )
            valence = st.slider(
                "Valence", 0.0, 1.0, 0.5, 0.01
            )
            tempo = st.slider(
                "Tempo (BPM)", 0.0, 250.0, 120.0, 1.0
            )

        submitted = st.form_submit_button("Find Cluster")

    if submitted:
        values = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo
        }

        input_data = pd.DataFrame(
            [values], columns=features
        )

        cluster = int(pipeline.predict(input_data)[0])

        st.success(f"Song group: {cluster_names[cluster]}")

        examples = songs[
            songs["cluster"] == cluster
        ]

        st.write("Example songs from this group")

        st.dataframe(
            examples[
                ["track_name", "artists"]
            ].sample(
                n=min(10, len(examples)),
                random_state=42
            ),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Examples share the predicted cluster; "
            "they are not ranked by similarity."
        )
