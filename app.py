import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import time

from model import predict_sentiment_proba

# ✅ PAGE CONFIG (BIG VISUAL UPGRADE 🔥)
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ✅ CUSTOM STYLING (VERY IMPORTANT 🔥)
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
}
.card {
    padding: 20px;
    border-radius: 10px;
    background-color: #1c1f26;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ✅ TITLE
st.markdown('<p class="big-title">📊 Phades Sentiment Analysis </p>', unsafe_allow_html=True)

# ✅ SESSION DATA
if "history" not in st.session_state:
    st.session_state.history = []

if "texts" not in st.session_state:
    st.session_state.texts = []

# -----------------------------
# ✅ INPUT SECTION (CARD STYLE)
# -----------------------------
st.markdown("### 🧠 Analyze Text")

user_input = st.text_area("Enter your text:")

col1, col2 = st.columns([1,1])

with col1:
    analyze = st.button("🔍 Analyze")

with col2:
    reset = st.button("♻️ Reset")

if analyze:
    if user_input.strip() != "":
        probs = predict_sentiment_proba(user_input)
        top_label = max(probs, key=probs.get)

        st.session_state.history.append(top_label)
        st.session_state.texts.append(user_input)

        st.success(f"✅ Prediction: {top_label}")

        # ✅ Probability bars (NEW 🔥)
        st.markdown("### 🔎 Confidence Levels")

        for label, score in probs.items():
            st.progress(float(score))
            st.write(f"{label}: {round(score*100,2)}%")

    else:
        st.warning("⚠️ Please enter text")

if reset:
    st.session_state.history = []
    st.session_state.texts = []
    st.warning("Data cleared")

time.sleep(0.3)
# -----------------------------
# ✅ LAYOUT: 2 COLUMNS
# -----------------------------
col1, col2 = st.columns(2)

# ✅ BAR CHART
with col1:
    st.markdown("### 📈 Sentiment Distribution")
    
if st.session_state.history:
    df_hist = pd.DataFrame({"sentiment": st.session_state.history})

    counts = df_hist["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        color="sentiment",
        animation_frame="count",
        title="Live Sentiment Count"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet")


# ✅ PIE CHART
with col2:
    st.markdown("### 🥧 Sentiment Breakdown")
    
if st.session_state.history:
    counts = pd.Series(st.session_state.history).value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig2 = px.pie(
        counts,
        names="sentiment",
        values="count",
        title="Sentiment Percentage",
        hole=0.4  # makes it donut style 🔥
    )

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# ✅ WORD CLOUD FULL WIDTH
# -----------------------------
st.markdown("### ☁️ Word Cloud")

if st.session_state.texts:
    combined_text = " ".join(st.session_state.texts)

    # ✅ EXTRA SAFETY CHECK
    
if st.session_state.texts:
    combined_text = " ".join(st.session_state.texts)

    if combined_text.strip() != "":
        # ✅ smaller size
        wc = WordCloud(
            width=300,
            height=150,
            background_color="black",
            colormap="viridis",
            max_words=50
        ).generate(combined_text)
        

        fig, ax = plt.subplots(figsize=(4, 2))  # ✅ controls display size
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        

        st.pyplot(fig)
    else:
        st.info("No valid text yet.")
else:
    st.info("Enter text to generate word cloud.")


# -----------------------------
# ✅ DOWNLOAD SECTION
# -----------------------------
st.markdown("### ⬇️ Export Data")

if st.session_state.history:
    df_hist_full = pd.DataFrame({
        "text": st.session_state.texts,
        "sentiment": st.session_state.history
    })

    csv = df_hist_full.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )