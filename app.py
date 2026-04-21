import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommender")

st.title("🎧 Music Recommender")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Required columns check
    required_cols = ['name', 'danceability', 'energy', 'tempo', 'valence']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain: name, danceability, energy, tempo, valence")
    else:
        df = df.dropna(subset=required_cols)
        df = df.sample(min(1000, len(df))).reset_index(drop=True)

        similarity = cosine_similarity(df[['danceability','energy','tempo','valence']])

        def is_hindi(song):
            return any(ord(c) > 127 for c in str(song))

        songs = df['name'].unique().tolist()

        playlist = st.multiselect("Select songs", songs)

        if st.button("Recommend"):
            if len(playlist) == 0:
                st.warning("Please select at least one song")
            else:
                indices = []
                for song in playlist:
                    match = df[df['name'] == song]
                    if not match.empty:
                        indices.append(match.index[0])

                if len(indices) == 0:
                    st.error("No matching songs found")
                else:
                    avg_similarity = np.mean(similarity[indices], axis=0)
                    scores = sorted(enumerate(avg_similarity), key=lambda x: x[1], reverse=True)

                    hindi_songs = []
                    english_songs = []

                    for i, score in scores:
                        name = df.iloc[i]['name']
                        if name in playlist:
                            continue

                        if is_hindi(name):
                            hindi_songs.append((name, score))
                        else:
                            english_songs.append((name, score))

                    final = []
                    final.extend(hindi_songs[:12])

                    remaining = 25 - len(final)
                    final.extend(english_songs[:remaining])

                    if len(final) < 25:
                        extra = hindi_songs[12:] + english_songs[remaining:]
                        final.extend(extra[:25 - len(final)])

                    final = final[:25]

                    rec_songs = [x[0] for x in final]
                    values = np.array([x[1] for x in final])

                    st.subheader("🎵 Recommended Songs")
                    for i, s in enumerate(rec_songs, 1):
                        st.write(f"{i}. {s}")

                    # Pie chart
                    if values.sum() != 0:
                        values = values / values.sum()

                        fig, ax = plt.subplots()
                        ax.pie(values, labels=rec_songs, autopct='%1.1f%%')
                        ax.set_title("Recommendation Distribution")

                        st.pyplot(fig)
