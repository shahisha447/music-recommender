import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Music Recommender")

st.title("🎧 Music Recommender")

# =========================
# LOAD CSV (from GitHub)
# =========================
try:
    df = pd.read_csv("data.csv")
except:
    st.error("❌ data.csv not found in GitHub repo")
    st.stop()

# =========================
# CLEAN DATA
# =========================
required_cols = ['name', 'danceability', 'energy', 'tempo', 'valence']

if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain: name, danceability, energy, tempo, valence")
    st.stop()

df = df.dropna(subset=required_cols)
df['name'] = df['name'].astype(str)

# reduce size safely
df = df.sample(min(1000, len(df))).reset_index(drop=True)

# =========================
# SIMILARITY
# =========================
features = ['danceability', 'energy', 'tempo', 'valence']
similarity = cosine_similarity(df[features])

# =========================
# SONG LIST
# =========================
songs = sorted(df['name'].unique().tolist())

# DEBUG (optional)
# st.write(songs)

# =========================
# UI
# =========================
playlist = st.multiselect("🎵 Select Songs", songs)

# =========================
# RECOMMEND BUTTON
# =========================
if st.button("Recommend 🎶"):

    if len(playlist) == 0:
        st.warning("⚠️ Please select at least one song")
    else:
        indices = []

        for song in playlist:
            match = df[df['name'] == song]
            if not match.empty:
                indices.append(match.index[0])

        if len(indices) == 0:
            st.error("❌ No matching songs found")
        else:
            avg_similarity = np.mean(similarity[indices], axis=0)
            scores = sorted(enumerate(avg_similarity), key=lambda x: x[1], reverse=True)

            # split songs
            hindi_songs = []
            english_songs = []

            def is_hindi(song):
                return any(ord(c) > 127 for c in song)

            for i, score in scores:
                name = df.iloc[i]['name']

                if name in playlist:
                    continue

                if is_hindi(name):
                    hindi_songs.append((name, score))
                else:
                    english_songs.append((name, score))

            # =========================
            # 25 SONG LOGIC
            # =========================
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

            # =========================
            # OUTPUT
            # =========================
            st.subheader("🎵 Recommended Songs")

            for i, s in enumerate(rec_songs, 1):
                st.write(f"{i}. {s}")

            # =========================
            # PIE CHART
            # =========================
            if values.sum() != 0:
                values = values / values.sum()

                fig, ax = plt.subplots()
                ax.pie(values, labels=rec_songs, autopct='%1.1f%%')
                ax.set_title("Recommendation Distribution")

                st.pyplot(fig)
