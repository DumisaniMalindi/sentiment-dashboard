import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import time

from model import predict_sentiment_proba

# ✅ PAGE CONFIG
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ✅ STYLE
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">📊 Phades Sentiment Analysis</p>', unsafe_allow_html=True)

# ✅ SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

if "texts" not in st.session_state:
    st.session_state.texts = []

# ✅ INPUT
st.markdown("### 🧠 Analyze Text")

user_input = st.text_area("Enter your text:", key="main_input")

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze", key="analyze_btn")

with col2:
    reset = st.button("♻️ Reset", key="reset_btn")

# ✅ ANALYZE
if analyze:
    if user_input.strip():
        probs = predict_sentiment_proba(user_input)
        top_label = max(probs, key=probs.get)

        st.session_state.history.append(top_label)
        st.session_state.texts.append(user_input)

        st.success(f"✅ Prediction: {top_label}")

        st.markdown("### 🔎 Confidence Levels")
        for label, score in probs.items():
            st.progress(float(score))
            st.write(f"{label}: {round(score*100, 2)}%")

    else:
        st.warning("⚠️ Please enter text")

# ✅ RESET
if reset:
    st.session_state.history = []
    st.session_state.texts = []
    st.warning("Data cleared")

time.sleep(0.2)

# ✅ CHARTS
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Sentiment Distribution")
    if st.session_state.history:
        df = pd.DataFrame({"sentiment": st.session_state.history})
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]

        fig = px.bar(counts, x="sentiment", y="count", color="sentiment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

with col2:
    st.markdown("### 🥧 Sentiment Breakdown")
    if st.session_state.history:
        counts = pd.Series(st.session_state.history).value_counts().reset_index()
        counts.columns = ["sentiment", "count"]

        fig2 = px.pie(counts, names="sentiment", values="count", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# ✅ WORD CLOUD
st.markdown("### ☁️ Word Cloud")

if st.session_state.texts:
    text = " ".join(st.session_state.texts)

    if text.strip():
        wc = WordCloud(width=400, height=200, background_color="black").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No valid text yet.")
else:
    st.info("Enter text to generate word cloud.")

# ✅ DOWNLOAD
st.markdown("### ⬇️ Export Data")

if st.session_state.history:
    df = pd.DataFrame({
        "text": st.session_state.texts,
        "sentiment": st.session_state.history
    })

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False),
        file_name="sentiment_results.csv",
        mime="text/csv"
    )